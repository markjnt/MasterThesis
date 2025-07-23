# -*- coding: utf-8 -*-
import os
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
import librosa 
import soundfile as sf
import subprocess
import argparse
import yaml
from types import SimpleNamespace
import platform
import time

# Importieren aller benötigten Module aus dem ursprünglichen Projekt
from loguru import logger
import smplx
from transformers import pipeline
from utils import config, logger_tools, other_tools_hf, other_tools, metric, data_transfer
from utils.joints import upper_body_mask, hands_body_mask, lower_body_mask
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from dataloaders.data_tools import joints_list
from utils import rotation_conversions as rc
from models.vq.model import RVQVAE

# --- Konfiguration und Initialisierung ---

# Überprüfen, ob eine GPU verfügbar ist, andernfalls CPU verwenden
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Umgebungsvariable für PyOpenGL auf Linux setzen
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Initialisierung des Whisper-Modells (wird bei erstem Gebrauch geladen)
asr_pipeline = None
def get_asr_pipeline():
    global asr_pipeline
    if asr_pipeline is None:
        logger.info("Initialisiere Whisper ASR-Modell (dies kann einen Moment dauern)...")
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny.en",
            chunk_length_s=30,
            device=device,
        )
    return asr_pipeline

debug = False

# --- Hilfsfunktionen zum Laden der Konfiguration ---

def nested_dict_to_namespace(d):
    if not isinstance(d, dict):
        return d
    namespace = SimpleNamespace()
    for key, value in d.items():
        setattr(namespace, key, nested_dict_to_namespace(value))
    return namespace

def load_config_from_yaml(config_path):
    logger.info(f"Lade Konfiguration von: {config_path}")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    args = nested_dict_to_namespace(config_dict)
    # Das cfg-Objekt wird aus dem model-Teil der Konfiguration erstellt
    cfg = nested_dict_to_namespace(config_dict.get('model', {}))
    # Um Kompatibilität zu gewährleisten, fügen wir cfg auch als Attribut zu args hinzu.
    args.model = cfg
    return args, cfg

# --- BaseTrainer Klasse (VOLLSTÄNDIG WIEDERHERGESTELLT) ---

class BaseTrainer(object):
    def __init__(self, args, cfg, ap):
        
        hf_dir = "hf"
        time_local = time.localtime()
        time_name_expend = "%02d%02d_%02d%02d%02d_"%(time_local[1], time_local[2],time_local[3], time_local[4], time_local[5])
        self.time_name_expend = time_name_expend
        tmp_dir = os.path.join(args.out_path, "custom", time_name_expend + hf_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        self.audio_path = os.path.join(tmp_dir, "tmp.wav")
        sf.write(self.audio_path, ap[1], ap[0])
        
        audio, ssr = librosa.load(self.audio_path,sr=args.audio_sr)
        
        # ASR-Modell verwenden, um Text-Transkripte zu erhalten
        file_path = os.path.join(tmp_dir, "tmp.lab")
        self.textgrid_path = os.path.join(tmp_dir, "tmp.TextGrid")
        if not debug:
            pipe = get_asr_pipeline()
            text = pipe(audio, batch_size=8)["text"]
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text)
            
            # Montreal Forced Aligner (MFA) verwenden
            logger.info("Führe Forced Alignment mit Montreal Forced Aligner (MFA) aus...")
            command = ["mfa", "align", tmp_dir, "english_us_arpa", "english_us_arpa", tmp_dir]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"MFA konnte nicht ausgeführt werden. Stellen Sie sicher, dass MFA korrekt installiert und im Systempfad ist.")
                logger.error(f"MFA Fehler: {result.stderr}")
                sys.exit(1)
            logger.info("MFA erfolgreich abgeschlossen.")

        self.args = args
        self.rank = 0 
        args.textgrid_file_path = self.textgrid_path
        args.audio_file_path = self.audio_path
        self.checkpoint_path = tmp_dir
        args.tmp_dir = tmp_dir
        
        if self.rank == 0:
            logger.info("Initialisiere Test-Dataloader...")
            self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test")
            self.test_loader = torch.utils.data.DataLoader(
                self.test_data, batch_size=1, shuffle=False, num_workers=args.loader_workers, drop_last=False
            )
        logger.info(f"Initialisierung des Test-Dataloaders erfolgreich.")
        
        model_module = __import__(f"models.{cfg.model_name}", fromlist=["something"])
        self.model = torch.nn.DataParallel(getattr(model_module, cfg.g_name)(cfg), args.gpus).to(device)
        
        if self.rank == 0:
            logger.info(self.model)
            logger.info(f"Initialisierung von {cfg.g_name} erfolgreich.")

        self.smplx = smplx.create(
            os.path.join(self.args.data_path_1, "smplx_models/"), model_type='smplx', gender='NEUTRAL_2020', 
            use_face_contour=False, num_betas=300, num_expression_coeffs=100, ext='npz', use_pca=False,
        ).to(self.rank).eval()    

        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list_face = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]
       
        self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        self.joints = 55
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1

        self.tracker = other_tools.EpochTracker(["fid", "l1div", "bc", "rec", "trans", "vel", "transv", 'dis', 'gen', 'acc', 'transa', 'exp', 'lvd', 'mse', "cls", "rec_face", "latent", "cls_full", "cls_self", "cls_word", "latent_word","latent_self","predict_x0_loss"], [False,True,True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,False, False, False,False,False,False])
        
        logger.info("Lade VQ-VAE-Modelle...")
        vq_model_module = __import__("models.motion_representation", fromlist=["something"])
        # self.vq_model_face = self._create_face_vq_model(vq_model_module) # Im Originalcode vorhanden, aber im Inferenz-Teil nicht genutzt.
        self.vq_models = self._create_body_vq_models()
        # self.vq_model_face.eval().to(self.rank)
        for model in self.vq_models.values():
            model.eval().to(self.rank)
        self.vq_model_upper, self.vq_model_hands, self.vq_model_lower = self.vq_models.values()
        self.vqvae_latent_scale = self.args.vqvae_latent_scale 
        self.args.vae_length = 240
        
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
        
        self.use_trans = self.args.use_trans
        self.mean = np.load(args.mean_pose_path)
        self.std = np.load(args.std_pose_path)
        
        for part in ['upper', 'hands', 'lower']:
            mask = globals()[f'{part}_body_mask']
            setattr(self, f'mean_{part}', torch.from_numpy(self.mean[mask]).to(device))
            setattr(self, f'std_{part}', torch.from_numpy(self.std[mask]).to(device))
        
        if self.args.use_trans:
            self.trans_mean = torch.from_numpy(np.load(self.args.mean_trans_path)).to(device)
            self.trans_std = torch.from_numpy(np.load(self.args.std_trans_path)).to(device)

    # Diese Funktion wurde im Originalcode nicht für die finale Inferenz verwendet
    # def _create_face_vq_model(self, module):
    #     self.args.vae_layer = 2
    #     self.args.vae_length = 256
    #     self.args.vae_test_dim = 106
    #     model = getattr(module, "VQVAEConvZero")(self.args).to(self.rank)
    #     other_tools.load_checkpoints(model, "./datasets/hub/pretrained_vq/face_vertex_1layer_790.bin", self.args.e_name)
    #     return model
    
    def _create_body_vq_models(self):
        vq_configs = {
            'upper': {'dim_pose': 78},
            'hands': {'dim_pose': 180},
            'lower': {'dim_pose': 54 if not self.args.use_trans else 57}
        }
        vq_models = {}
        for part, config in vq_configs.items():
            model = self._create_rvqvae_model(config['dim_pose'], part)
            vq_models[part] = model
        return vq_models
    
    def _create_rvqvae_model(self, dim_pose: int, body_part: str) -> RVQVAE:
        args = self.args
        model = RVQVAE(
            args, dim_pose, args.nb_code, args.code_dim, args.code_dim,
            args.down_t, args.stride_t, args.width, args.depth,
            args.dilation_growth_rate, args.vq_act, args.vq_norm
        )
        checkpoint_path = getattr(args, f'vqvae_{body_part}_path')
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)['net'])
        return model
      
    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).to(device)
        original_shape_t = torch.zeros((n, 165), device=device)
        selected_indices = torch.where(selection_array == 1)[0]
        # Ensure filtered_t is correctly shaped
        original_shape_t[:, selected_indices] = filtered_t.reshape(n, -1)
        return original_shape_t

    def _load_data(self, dict_data):
        tar_pose_raw = dict_data["pose"]
        tar_pose = tar_pose_raw[:, :, :165].to(self.rank)
        tar_contact = tar_pose_raw[:, :, 165:169].to(self.rank)
        tar_trans = dict_data["trans"].to(self.rank)
        tar_trans_v = dict_data["trans_v"].to(self.rank)
        tar_exps = dict_data["facial"].to(self.rank)
        in_audio = dict_data["audio"].to(self.rank)
        if 'wavlm' in dict_data:
            wavlm = dict_data["wavlm"].to(self.rank)
        else:
            wavlm = None
        in_word = dict_data["word"].to(self.rank)
        tar_beta = dict_data["beta"].to(self.rank)
        tar_id = dict_data["id"].to(self.rank).long()
        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

        tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)

        tar_pose_lower = tar_pose_leg

        if self.args.pose_norm:
            tar_pose_upper = (tar_pose_upper - self.mean_upper) / self.std_upper
            tar_pose_hands = (tar_pose_hands - self.mean_hands) / self.std_hands
            tar_pose_lower = (tar_pose_lower - self.mean_lower) / self.std_lower
        
        if self.use_trans:
            tar_trans_v = (tar_trans_v - self.trans_mean)/self.trans_std
            tar_pose_lower = torch.cat([tar_pose_lower,tar_trans_v], dim=-1)
      
        latent_upper_top = self.vq_model_upper.map2latent(tar_pose_upper)
        latent_hands_top = self.vq_model_hands.map2latent(tar_pose_hands)
        latent_lower_top = self.vq_model_lower.map2latent(tar_pose_lower)
        
        latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2)/self.args.vqvae_latent_scale
        
        style_feature = None
        
        return {
            "in_audio": in_audio, "wavlm": wavlm, "in_word": in_word, "tar_trans": tar_trans,
            "tar_exps": tar_exps, "tar_beta": tar_beta, "tar_pose": tar_pose, "latent_in":  latent_in,
            "tar_id": tar_id, "tar_contact": tar_contact, "style_feature": style_feature,
        }

    def _g_test(self, loaded_data):
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        tar_pose = loaded_data["tar_pose"]
        tar_trans = loaded_data["tar_trans"]
        in_word = loaded_data["in_word"]
        in_audio = loaded_data["in_audio"]
        in_seed = loaded_data['latent_in']
        
        remain = n % 8
        if remain != 0:
            n = n - remain
            tar_pose = tar_pose[:, :-remain, :]
            tar_trans = tar_trans[:, :-remain, :]
            in_word = in_word[:, :-remain]
            in_audio = in_audio[:, :n * (16000//30)]
            in_seed = in_seed[:, :in_seed.shape[1]-(remain//self.args.vqvae_squeeze_scale), :]

        rec_all_upper, rec_all_hands, rec_all_lower = [], [], []
        vqvae_squeeze_scale = self.args.vqvae_squeeze_scale
        roundt = (n - self.args.pre_frames * vqvae_squeeze_scale) // (self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale)
        round_l = self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale
         
        last_sample = None
        for i in range(roundt):
            in_word_tmp = in_word[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames * vqvae_squeeze_scale]
            in_audio_tmp = in_audio[:, i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*self.args.pre_frames * vqvae_squeeze_scale]
            in_id_tmp = loaded_data['tar_id'][:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            in_seed_tmp_part = in_seed[:, (i+1)*(round_l)//vqvae_squeeze_scale : (i+1)*(round_l)//vqvae_squeeze_scale+self.args.pre_frames]

            if i == 0:
                in_seed_tmp = in_seed[:, 0 : self.args.pose_length//vqvae_squeeze_scale]
            else:
                in_seed_tmp = torch.cat([last_sample[:, -self.args.pre_frames:, :], in_seed_tmp_part], dim=1)

            cond_ = {'y':{}}
            cond_['y']['audio'] = in_audio_tmp
            cond_['y']['word'] = in_word_tmp
            cond_['y']['id'] = in_id_tmp
            cond_['y']['seed'] = in_seed_tmp
            cond_['y']['mask'] = (torch.zeros([self.args.batch_size, 1, 1, self.args.pose_length]) < 1).to(device)
            cond_['y']['style_feature'] = torch.zeros([bs, 512]).to(device)

            sample = self.model(cond_)['latents'].squeeze(2).permute(0, 2, 1)
            last_sample = sample.clone()
            
            rec_latent_upper, rec_latent_hands, rec_latent_lower = sample.split(128, dim=-1)
            
            start_idx = self.args.pre_frames if i > 0 else 0
            rec_all_upper.append(rec_latent_upper[:, start_idx:])
            rec_all_hands.append(rec_latent_hands[:, start_idx:])
            rec_all_lower.append(rec_latent_lower[:, start_idx:])

        rec_all_upper = torch.cat(rec_all_upper, dim=1) * self.vqvae_latent_scale
        rec_all_hands = torch.cat(rec_all_hands, dim=1) * self.vqvae_latent_scale
        rec_all_lower = torch.cat(rec_all_lower, dim=1) * self.vqvae_latent_scale

        rec_upper = self.vq_model_upper.latent2origin(rec_all_upper)[0]
        rec_hands = self.vq_model_hands.latent2origin(rec_all_hands)[0]
        rec_lower = self.vq_model_lower.latent2origin(rec_all_lower)[0]
        
        rec_trans = torch.zeros_like(tar_trans)
        if self.use_trans:
            rec_trans_v = rec_lower[...,-3:] * self.trans_std + self.trans_mean
            # Integration to get absolute translation
            rec_trans = torch.cumsum(rec_trans_v, dim=1)
            rec_lower = rec_lower[...,:-3]
        
        if self.args.pose_norm:
            rec_upper = rec_upper * self.std_upper + self.mean_upper
            rec_hands = rec_hands * self.std_hands + self.mean_hands
            rec_lower = rec_lower * self.std_lower + self.mean_lower

        n_rec = rec_lower.shape[1]
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_upper.reshape(bs, n_rec, 13, 6))
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n_rec, 13*3)
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n_rec)
        
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_lower.reshape(bs, n_rec, 9, 6))
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n_rec, 9*3)
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n_rec)
        
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_hands.reshape(bs, n_rec, 30, 6))
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n_rec, 30*3)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n_rec)
        
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
        rec_pose[:, 66:69] = tar_pose.reshape(-1, 165)[:bs*n_rec, 66:69]

        # Konvertiere zurück in 6D für Konsistenz, obwohl Achsenwinkel für NPZ gespeichert wird
        rec_pose_6d = rc.matrix_to_rotation_6d(rc.axis_angle_to_matrix(rec_pose.reshape(bs*n_rec, j, 3))).reshape(bs, n_rec, j*6)
        
        return {
            'rec_pose': rec_pose_6d,
            'rec_trans': rec_trans[:,:n_rec,:],
            'rec_exps': loaded_data['tar_exps'][:, :n_rec, :],
            'tar_beta': loaded_data['tar_beta'][:, :n_rec, :],
        }

    def starte_inferenz_und_speichere_npz(self):
        results_save_path = os.path.join(self.checkpoint_path, "output")
        os.makedirs(results_save_path, exist_ok=True)
        
        logger.info("Starte Inferenzprozess...")
        start_time = time.time()
        self.model.eval()
        self.smplx.eval()

        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_test(loaded_data)
                
                rec_pose_6d = net_out['rec_pose']
                rec_trans = net_out['rec_trans']
                rec_exps = net_out['rec_exps']
                
                bs, n, _ = rec_pose_6d.shape
                j = self.joints
                
                # Konvertiere finale 6D-Pose in Achsenwinkel für die Speicherung in NPZ
                rec_pose_matrix = rc.rotation_6d_to_matrix(rec_pose_6d.reshape(bs*n, j, 6))
                rec_pose_axis_angle = rc.matrix_to_axis_angle(rec_pose_matrix).reshape(bs*n, j*3)

                rec_pose_np = rec_pose_axis_angle.detach().cpu().numpy()
                rec_trans_np = rec_trans.reshape(bs*n, 3).detach().cpu().numpy()
                rec_exp_np = rec_exps.reshape(bs*n, 100).detach().cpu().numpy()
                
                # Verwende Betas aus den geladenen Daten oder lade eine Beispieldatei
                try:
                    betas_np = net_out['tar_beta'].reshape(-1, 300).detach().cpu().numpy()[0]
                except:
                    gt_npz = np.load("./demo/examples/2_scott_0_1_1.npz", allow_pickle=True)
                    betas_np = gt_npz["betas"]

                results_npz_file_save_path = os.path.join(results_save_path, f"ergebnis_{self.time_name_expend}.npz")
                
                np.savez(results_npz_file_save_path,
                    betas=betas_np,
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate=self.args.pose_fps,
                )

                end_time = time.time() - start_time
                total_length_s = n / self.args.pose_fps
                logger.info(f"Gesamte Inferenzzeit: {end_time:.2f} s für {total_length_s:.2f} s Bewegung.")
                
                return results_npz_file_save_path
        return None

# --- Haupt-Orchestrierungsfunktion ---

@logger.catch
def erzeuge_bewegung_aus_audio(args, cfg, input_audio_pfad: str) -> str:
    samplerate, audio_data = sf.read(input_audio_pfad)
    audio_paket = (samplerate, audio_data)

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    other_tools_hf.set_random_seed(args)
    
    trainer = BaseTrainer(args, cfg, ap=audio_paket)
    
    logger.info(f"Lade Checkpoint von: {args.test_ckpt}")
    other_tools.load_checkpoints(trainer.model, args.test_ckpt, args.g_name)
    
    npz_datei_pfad = trainer.starte_inferenz_und_speichere_npz()
    return npz_datei_pfad

# --- Hauptausführungsblock ---
if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = '127.0.0.3'
    os.environ["MASTER_PORT"] = '8678'
    
    parser = argparse.ArgumentParser(description="Erzeuge 3D-Bewegungen aus einer Audiodatei mit einer Konfigurationsdatei.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Pfad zur config.yaml Konfigurationsdatei.")
    parser.add_argument("--audio", type=str, required=True, help="Pfad zur Eingabe-Audiodatei (WAV).")
    cli_args = parser.parse_args()

    if not os.path.exists(cli_args.config):
        logger.error(f"Konfigurationsdatei nicht gefunden unter: {cli_args.config}")
        sys.exit(1)
        
    args, cfg = load_config_from_yaml(cli_args.config)
    
    if not os.path.exists(cli_args.audio):
        logger.error(f"Audiodatei nicht gefunden unter: {cli_args.audio}")
        sys.exit(1)

    logger.info(f"Verarbeite Audiodatei: {cli_args.audio}")
    
    output_npz_pfad = erzeuge_bewegung_aus_audio(args, cfg, cli_args.audio)
    
    if output_npz_pfad:
        logger.success(f"Bewegungsdatei erfolgreich erstellt unter: {output_npz_pfad}")
    else:
        logger.error("Fehler bei der Erstellung der Bewegungsdatei.")