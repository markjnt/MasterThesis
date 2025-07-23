# -*- coding: utf-8 -*-
import os
import signal
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np
import pprint
from loguru import logger
import smplx
from torch.utils.tensorboard import SummaryWriter
import wandb
import matplotlib.pyplot as plt
from utils import config, logger_tools, other_tools_hf, metric, data_transfer, other_tools
from utils.joints import upper_body_mask, hands_body_mask, lower_body_mask
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from dataloaders.data_tools import joints_list
from utils import rotation_conversions as rc
import soundfile as sf
import librosa 
import subprocess
from transformers import pipeline
from models.vq.model import RVQVAE

# Überprüfen, ob eine GPU verfügbar ist, andernfalls CPU verwenden
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Umgebungsvariable für PyOpenGL auf Linux setzen
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Initialisierung des Whisper-Modells für die Spracherkennung
pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-tiny.en",
  chunk_length_s=30,
  device=device,
)       

debug = False

class BaseTrainer(object):
    def __init__(self, args, cfg, audio_paket):
        """
        Initialisiert den Trainer. Diese Methode richtet temporäre Verzeichnisse ein,
        verarbeitet die Eingabe-Audiodatei, führt Spracherkennung (ASR) und Forced Alignment (MFA) durch
        und lädt alle notwendigen Modelle (Gestenerzeugung, VQ-VAE, SMPLX).
        """
        hf_dir = "hf"
        time_local = time.localtime()
        time_name_expend = "%02d%02d_%02d%02d%02d_"%(time_local[1], time_local[2],time_local[3], time_local[4], time_local[5])
        self.time_name_expend = time_name_expend
        # Ein temporäres Verzeichnis für die Verarbeitung erstellen
        tmp_dir = args.out_path + "custom/"+ time_name_expend + hf_dir
        if not os.path.exists(tmp_dir + "/"):
            os.makedirs(tmp_dir + "/")
        self.audio_path = tmp_dir + "/tmp.wav"
        sf.write(self.audio_path, audio_paket[1], audio_paket[0])
        
        audio, ssr = librosa.load(self.audio_path,sr=args.audio_sr)
        
        # ASR-Modell verwenden, um Text-Transkripte zu erhalten
        file_path = tmp_dir+"/tmp.lab"
        self.textgrid_path = tmp_dir + "/tmp.TextGrid"
        if not debug:
            logger.info("Führe Spracherkennung (ASR) mit Whisper aus...")
            text = pipe(audio, batch_size=8)["text"]
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text)
            
            # Montreal Forced Aligner (MFA) verwenden, um TextGrid zu erhalten
            logger.info("Führe Forced Alignment mit MFA aus...")
            # HINWEIS: MFA muss in Ihrer Umgebung installiert und konfiguriert sein.
            command = ["mfa", "align", tmp_dir, "english_us_arpa", "english_us_arpa", tmp_dir]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            logger.info(f"MFA-Ausgabe: {result.stdout}")
            if result.stderr:
                logger.error(f"MFA-Fehler: {result.stderr}")

        self.args = args
        self.rank = 0 
       
        args.textgrid_file_path = self.textgrid_path
        args.audio_file_path = self.audio_path
        self.checkpoint_path = tmp_dir
        args.tmp_dir = tmp_dir
        
        # DataLoader für die benutzerdefinierte Eingabe initialisieren
        logger.info("Initialisiere Test-Dataloader...")
        self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test")
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data, batch_size=1, shuffle=False, num_workers=args.loader_workers, drop_last=False
        )
        logger.info("Initialisierung des Test-Dataloaders erfolgreich.")
        
        # Haupt-Gestenerzeugungsmodell laden
        model_module = __import__(f"models.{cfg.model.model_name}", fromlist=["something"])
        self.model = torch.nn.DataParallel(getattr(model_module, cfg.model.g_name)(cfg), args.gpus).cuda()
        logger.info(f"Initialisierung von {cfg.model.g_name} erfolgreich.")

        # SMPLX-Modell laden
        self.smplx = smplx.create(
            self.args.data_path_1+"smplx_models/", model_type='smplx', gender='NEUTRAL_2020', 
            use_face_contour=False, num_betas=300, num_expression_coeffs=100, ext='npz', use_pca=False,
        ).to(self.rank).eval()    

        # Gelenkmasken und Normalisierungsdaten vorbereiten
        self._prepare_joints_and_normalization()

        # VQ-VAE-Modelle für verschiedene Körperteile laden
        self._load_vq_vae_models()
        
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
    
    def _prepare_joints_and_normalization(self):
        """Bereitet Gelenkmasken und Normalisierungsstatistiken vor."""
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list_face = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]
       
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_upper:
            start, end = self.ori_joint_list[joint_name]
            self.joint_mask_upper[start:end] = 1
        
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_hands:
            start, end = self.ori_joint_list[joint_name]
            self.joint_mask_hands[start:end] = 1
            
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_lower:
            start, end = self.ori_joint_list[joint_name]
            self.joint_mask_lower[start:end] = 1

        self.joints = 55
        self.use_trans = self.args.use_trans
        mean = np.load(self.args.mean_pose_path)
        std = np.load(self.args.std_pose_path)
        
        for part in ['upper', 'hands', 'lower']:
            mask = globals()[f'{part}_body_mask']
            setattr(self, f'mean_{part}', torch.from_numpy(mean[mask]).cuda())
            setattr(self, f'std_{part}', torch.from_numpy(std[mask]).cuda())
        
        if self.args.use_trans:
            self.trans_mean = torch.from_numpy(np.load(self.args.mean_trans_path)).cuda()
            self.trans_std = torch.from_numpy(np.load(self.args.std_trans_path)).cuda()

    def _load_vq_vae_models(self):
        """Initialisiert und lädt die vortrainierten VQ-VAE-Modelle."""
        logger.info("Lade VQ-VAE-Modelle...")
        vq_model_module = __import__("models.motion_representation", fromlist=["something"])
        
        # Laden der Körperteil-VQ-Modelle
        self.vq_models = self._create_body_vq_models()
        for model in self.vq_models.values():
            model.eval().to(self.rank)
        self.vq_model_upper, self.vq_model_hands, self.vq_model_lower = self.vq_models.values()
        
        self.vqvae_latent_scale = self.args.vqvae_latent_scale 
        self.args.vae_length = 240
        logger.info("VQ-VAE-Modelle erfolgreich geladen.")

    def _create_body_vq_models(self):
        """Erstellt VQ-VAE-Modelle für Oberkörper, Hände und Unterkörper."""
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
        """Erstellt ein einzelnes RVQVAE-Modell und lädt die Gewichte."""
        args = self.args
        model = RVQVAE(
            args, dim_pose, args.nb_code, args.code_dim, args.code_dim,
            args.down_t, args.stride_t, args.width, args.depth,
            args.dilation_growth_rate, args.vq_act, args.vq_norm
        )
        checkpoint_path = getattr(args, f'vqvae_{body_part}_path')
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)['net'])
        return model

    # Die restlichen Methoden der Klasse (`inverse_selection_tensor`, `_load_data`, `_g_test`) bleiben
    # im Wesentlichen unverändert gegenüber dem Originalskript. Hier werden sie der Vollständigkeit halber eingefügt.
    # ... (Code für inverse_selection_tensor, _load_data, _g_test hier einfügen) ...
    #
    # Der Einfachheit halber werden die Methoden hier nicht erneut vollständig abgedruckt,
    # da sie in der ursprünglichen Frage bereits vorhanden sind. Es ist wichtig, sie hierher zu kopieren.
    # Wir nehmen an, dass sie hier sind und erstellen nur die neue, vereinfachte Ausführungsmethode.
    
    # Platzhalter: Bitte fügen Sie die Methoden _load_data und _g_test aus dem Originalskript hier ein.
    # Wichtig: Die Methoden müssen unverändert aus dem Originalskript übernommen werden.
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).to(self.rank)
        original_shape_t = torch.zeros((n, 165)).to(self.rank)
        selected_indices = torch.where(selection_array == 1)[0]
        original_shape_t[:, selected_indices] = filtered_t
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
            "tar_exps": tar_exps, "tar_beta": tar_beta, "tar_pose": tar_pose, "latent_in": latent_in,
            "tar_id": tar_id, "tar_contact": tar_contact, "style_feature": style_feature,
        }

    def _g_test(self, loaded_data):
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints
        in_audio = loaded_data["in_audio"]
        in_word = loaded_data["in_word"]
        in_seed = loaded_data['latent_in']
        
        remain = n % 8
        if remain != 0:
            n = n - remain
            in_word = in_word[:, :-remain]
            in_seed = in_seed[:, :in_seed.shape[1] - (remain // self.args.vqvae_squeeze_scale), :]

        rec_all_upper, rec_all_hands, rec_all_lower = [], [], []
        vqvae_squeeze_scale = self.args.vqvae_squeeze_scale
        round_l = self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale
        roundt = (n - self.args.pre_frames * vqvae_squeeze_scale) // round_l
        
        last_sample = None
        for i in range(roundt):
            audio_start = i * (16000 // 30 * round_l)
            audio_end = (i + 1) * (16000 // 30 * round_l) + 16000 // 30 * self.args.pre_frames * vqvae_squeeze_scale
            in_audio_tmp = in_audio[:, audio_start:audio_end]
            
            seed_start = i * (round_l // vqvae_squeeze_scale)
            seed_end = (i + 1) * (round_l // vqvae_squeeze_scale) + self.args.pre_frames
            
            if i == 0:
                in_seed_tmp = in_seed[:, seed_start:seed_end]
            else:
                in_seed_tmp = torch.cat([last_sample[:, -self.args.pre_frames:, :], in_seed[:, seed_end-self.args.pre_frames:seed_end]], dim=1)


            cond_ = {'y': {
                'audio': in_audio_tmp,
                'word': in_word[:, i * round_l:(i + 1) * round_l + self.args.pre_frames * vqvae_squeeze_scale],
                'id': loaded_data['tar_id'][:, i * round_l:(i + 1) * round_l + self.args.pre_frames],
                'seed': in_seed_tmp,
                'mask': (torch.zeros([self.args.batch_size, 1, 1, self.args.pose_length]) < 1).to(self.rank),
                'style_feature': torch.zeros([bs, 512]).to(self.rank),
            }}
            
            sample = self.model(cond_)['latents'].squeeze().permute(1, 0).unsqueeze(0)
            last_sample = sample.clone()
            
            rec_latent_upper = sample[..., :128]
            rec_latent_hands = sample[..., 128:2*128]
            rec_latent_lower = sample[..., 2*128:]
            
            if i == 0:
                rec_all_upper.append(rec_latent_upper)
                rec_all_hands.append(rec_latent_hands)
                rec_all_lower.append(rec_latent_lower)
            else:
                rec_all_upper.append(rec_latent_upper[:, self.args.pre_frames:])
                rec_all_hands.append(rec_latent_hands[:, self.args.pre_frames:])
                rec_all_lower.append(rec_latent_lower[:, self.args.pre_frames:])

        rec_all_upper = torch.cat(rec_all_upper, dim=1) * self.vqvae_latent_scale
        rec_all_hands = torch.cat(rec_all_hands, dim=1) * self.vqvae_latent_scale
        rec_all_lower = torch.cat(rec_all_lower, dim=1) * self.vqvae_latent_scale

        rec_upper = self.vq_model_upper.latent2origin(rec_all_upper)[0]
        rec_hands = self.vq_model_hands.latent2origin(rec_all_hands)[0]
        rec_lower = self.vq_model_lower.latent2origin(rec_all_lower)[0]
        
        rec_trans = torch.zeros((bs, rec_lower.shape[1], 3), device=self.rank)
        if self.use_trans:
            rec_trans_v = rec_lower[..., -3:] * self.trans_std + self.trans_mean
            rec_trans = torch.cumsum(rec_trans_v, dim=1)
            rec_lower = rec_lower[..., :-3]
        
        if self.args.pose_norm:
            rec_upper = rec_upper * self.std_upper + self.mean_upper
            rec_hands = rec_hands * self.std_hands + self.mean_hands
            rec_lower = rec_lower * self.std_lower + self.mean_lower

        n = rec_lower.shape[1]
        rec_pose_upper = rc.matrix_to_axis_angle(rc.rotation_6d_to_matrix(rec_upper.reshape(bs, n, 13, 6))).reshape(bs * n, 13 * 3)
        rec_pose_lower = rc.matrix_to_axis_angle(rc.rotation_6d_to_matrix(rec_lower.reshape(bs, n, 9, 6))).reshape(bs * n, 9 * 3)
        rec_pose_hands = rc.matrix_to_axis_angle(rc.rotation_6d_to_matrix(rec_hands.reshape(bs, n, 30, 6))).reshape(bs * n, 30 * 3)
        
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
        
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
        rec_pose[:, 66:69] = loaded_data["tar_pose"].reshape(bs*loaded_data["tar_pose"].shape[1], -1)[:bs*n, 66:69]

        rec_pose_6d = rc.matrix_to_rotation_6d(rc.axis_angle_to_matrix(rec_pose.reshape(bs*n, j, 3))).reshape(bs, n, j*6)
        
        return {
            'rec_pose': rec_pose_6d,
            'rec_trans': rec_trans,
            'rec_exps': loaded_data['tar_exps'][:, :n, :], # Verwende die Expressionen aus den Originaldaten als Platzhalter
        }

    def starte_inferenz_und_speichere_npz(self):
        '''
        Vereinfachte Methode, um die Inferenz durchzuführen und die resultierende Bewegung
        in einer NPZ-Datei zu speichern. Gibt den Pfad zur NPZ-Datei zurück.
        '''
        results_save_path = self.checkpoint_path + "/output/"
        if not os.path.exists(results_save_path):
            os.makedirs(results_save_path)
        
        logger.info("Starte Inferenzprozess...")
        start_time = time.time()
        self.model.eval()
        self.smplx.eval()

        with torch.no_grad():
            # Der Dataloader enthält nur ein Element (unsere Audiodatei), daher läuft die Schleife nur einmal.
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_test(loaded_data)
                
                rec_pose = net_out['rec_pose']
                rec_trans = net_out['rec_trans']
                rec_exps = net_out['rec_exps']
                
                bs, n, j = rec_pose.shape[0], rec_pose.shape[1], self.joints
                
                # Konvertiere die 6D-Rotationsdarstellung in Achsenwinkel für die Speicherung
                rec_pose_matrix = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose_axis_angle = rc.matrix_to_axis_angle(rec_pose_matrix).reshape(bs*n, j*3)

                # Konvertiere Tensoren in Numpy-Arrays
                rec_pose_np = rec_pose_axis_angle.detach().cpu().numpy()
                rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs*n, 3)
                rec_exp_np = rec_exps.detach().cpu().numpy().reshape(bs*n, 100) 
                
                # Lade eine Beispiel-NPZ, um die 'betas' (Körperformparameter) zu erhalten.
                # Das Modell generiert diese nicht, daher werden hier feste Werte verwendet.
                gt_npz = np.load("./demo/examples/2_scott_0_1_1.npz", allow_pickle=True)

                # Definiere den Speicherpfad für die Ergebnis-NPZ-Datei
                results_npz_file_save_path = results_save_path+f"ergebnis_{self.time_name_expend}"+'.npz'
                
                np.savez(results_npz_file_save_path,
                    betas=gt_npz["betas"],  # Feste Betas aus einer Beispieldatei
                    poses=rec_pose_np,
                    expressions=rec_exp_np, # Verwendet Platzhalter-Gesichtsanimationen
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )

                end_time = time.time() - start_time
                total_length_s = rec_pose_np.shape[0] / 30
                logger.info(f"Gesamte Inferenzzeit: {end_time:.2f} s für {total_length_s:.2f} s Bewegung.")
                
                # Da die Schleife nur einmal läuft, geben wir den Pfad direkt zurück.
                return results_npz_file_save_path
        
        return None # Sollte nicht erreicht werden, wenn Daten vorhanden sind

@logger.catch
def erzeuge_bewegung_aus_audio(input_audio_pfad: str) -> str:
    """
    Hauptfunktion zur Orchestrierung des Prozesses.
    Nimmt einen Audiopfad entgegen und gibt den Pfad zur generierten NPZ-Datei zurück.

    Args:
        input_audio_pfad (str): Der Pfad zur WAV-Audiodatei.

    Returns:
        str: Der Pfad zur generierten NPZ-Bewegungsdatei.
    """
    args, cfg = config.parse_args()
    
    # Lade die Audiodatei in das vom Trainer erwartete Format (samplerate, data_array)
    samplerate, audio_data = sf.read(input_audio_pfad)
    audio_paket = (samplerate, audio_data)

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    other_tools_hf.set_random_seed(args)
    other_tools_hf.print_exp_info(args)

    # Initialisiere den Trainer mit den Konfigurationen und der Audiodatei
    trainer = BaseTrainer(args, cfg, audio_paket=audio_paket)
    
    # Lade den vortrainierten Modell-Checkpoint
    logger.info(f"Lade Checkpoint von: {args.test_ckpt}")
    other_tools.load_checkpoints(trainer.model, args.test_ckpt, args.g_name)
    
    # Starte den Inferenzprozess und erhalte den Pfad zur NPZ-Datei
    npz_datei_pfad = trainer.starte_inferenz_und_speichere_npz()
    
    return npz_datei_pfad

# ---------- HAUPTAUSFÜHRUNGSBLOCK ----------
if __name__ == "__main__":
    # Setzen der Umgebungsvariablen, die vom Skript benötigt werden
    os.environ["MASTER_ADDR"]='127.0.0.3'
    os.environ["MASTER_PORT"]='8678'
    
    # Pfad zur Eingabe-Audiodatei. Ändern Sie dies zu Ihrer gewünschten Datei.
    # WICHTIG: Es wird erwartet, dass es sich um eine englische Audiodatei handelt,
    # da die ASR- und Aligner-Modelle für Englisch konfiguriert sind.
    audio_datei = "demo/examples/2_scott_0_1_1.wav"
    
    if not os.path.exists(audio_datei):
        logger.error(f"Audiodatei nicht gefunden unter: {audio_datei}")
        sys.exit(1)

    logger.info(f"Verarbeite Audiodatei: {audio_datei}")
    
    # Rufe die Hauptfunktion auf, um die Bewegung zu erzeugen
    output_npz_pfad = erzeuge_bewegung_aus_audio(audio_datei)
    
    if output_npz_pfad:
        logger.info(f"Bewegungsdatei erfolgreich erstellt unter: {output_npz_pfad}")
    else:
        logger.error("Fehler bei der Erstellung der Bewegungsdatei.")