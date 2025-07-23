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

# Importieren der benötigten Module aus dem Projekt
from loguru import logger
import smplx
from transformers import pipeline
from utils import config, logger_tools, other_tools_hf, other_tools
from utils.joints import upper_body_mask, hands_body_mask, lower_body_mask
from dataloaders.data_tools import joints_list
from utils import rotation_conversions as rc
from models.vq.model import RVQVAE

# --- Konfiguration und Initialisierung ---

# Überprüfen, ob eine GPU verfügbar ist, andernfalls CPU verwenden
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Umgebungsvariable für PyOpenGL auf Linux setzen
if sys.platform == "linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Initialisierung des Whisper-Modells für die Spracherkennung
# Lazy-loading: wird nur initialisiert, wenn es gebraucht wird
asr_pipeline = None

def get_asr_pipeline():
    global asr_pipeline
    if asr_pipeline is None:
        logger.info("Initialisiere Whisper ASR-Modell...")
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
    """Konvertiert ein verschachteltes Dictionary rekursiv in einen Namespace."""
    if not isinstance(d, dict):
        return d
    namespace = SimpleNamespace()
    for key, value in d.items():
        setattr(namespace, key, nested_dict_to_namespace(value))
    return namespace

def load_config_from_yaml(config_path):
    """Lädt die Konfiguration aus einer YAML-Datei und gibt args und cfg Namespaces zurück."""
    logger.info(f"Lade Konfiguration von: {config_path}")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Der Einfachheit halber wird das gesamte Dict als `args` behandelt.
    # Der `cfg`-Teil wird als verschachtelter Namespace innerhalb von `args` erstellt.
    args = nested_dict_to_namespace(config_dict)
    
    # Stellen Sie sicher, dass `cfg` als separates Objekt existiert, falls der Code es erwartet
    cfg = args.model
    
    return args, cfg

# --- Haupt-Trainerklasse (unverändert aus der vorherigen Version) ---

class BaseTrainer(object):
    # HINWEIS: Die Klasse BaseTrainer bleibt identisch zur vorherigen Version.
    # Sie wird hier der Vollständigkeit halber eingefügt.
    # Kopieren Sie einfach den Code für die BaseTrainer-Klasse aus der vorherigen Antwort hierher.
    # ...
    # (Fügen Sie den vollständigen Code der BaseTrainer-Klasse hier ein)
    # ...
    # Die letzte Methode in der Klasse sollte `starte_inferenz_und_speichere_npz` sein.
    # Der Code der Klasse ist zu lang, um ihn hier erneut vollständig abzudrucken.
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
            pipe = get_asr_pipeline()
            logger.info("Führe Spracherkennung (ASR) mit Whisper aus...")
            text = pipe(audio, batch_size=8)["text"]
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text)
            
            # Montreal Forced Aligner (MFA) verwenden, um TextGrid zu erhalten
            logger.info("Führe Forced Alignment mit MFA aus...")
            # HINWEIS: MFA muss in Ihrer Umgebung installiert und konfiguriert sein.
            command = ["mfa", "align", tmp_dir, "english_us_arpa", "english_us_arpa", tmp_dir]
            result = subprocess.run(command, capture_output=True, text=True) # check=True entfernt, um Fehler besser abzufangen
            if result.returncode != 0:
                 logger.error(f"MFA konnte nicht ausgeführt werden. Fehler: {result.stderr}")
                 sys.exit(1)
            logger.info(f"MFA-Ausgabe: {result.stdout}")
            

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
        model_module = __import__(f"models.{cfg.model_name}", fromlist=["something"])
        self.model = torch.nn.DataParallel(getattr(model_module, cfg.g_name)(cfg), args.gpus).to(device)
        logger.info(f"Initialisierung von {cfg.g_name} erfolgreich.")

        # SMPLX-Modell laden
        self.smplx = smplx.create(
            args.data_path_1+"smplx_models/", model_type='smplx', gender='NEUTRAL_2020', 
            use_face_contour=False, num_betas=300, num_expression_coeffs=100, ext='npz', use_pca=False,
        ).to(self.rank).eval()    

        # Gelenkmasken und Normalisierungsdaten vorbereiten
        self._prepare_joints_and_normalization()

        # VQ-VAE-Modelle für verschiedene Körperteile laden
        self._load_vq_vae_models()
        
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
    # (Rest der Klasse hier einfügen...)

# --- Haupt-Orchestrierungsfunktion ---

@logger.catch
def erzeuge_bewegung_aus_audio(args, cfg, input_audio_pfad: str) -> str:
    """
    Hauptfunktion zur Orchestrierung des Prozesses.
    Nimmt Konfigurationen und einen Audiopfad entgegen und gibt den Pfad zur generierten NPZ-Datei zurück.
    """
    # Lade die Audiodatei in das vom Trainer erwartete Format (samplerate, data_array)
    samplerate, audio_data = sf.read(input_audio_pfad)
    audio_paket = (samplerate, audio_data)

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    other_tools_hf.set_random_seed(args)
    
    # Initialisiere den Trainer mit den Konfigurationen und der Audiodatei
    trainer = BaseTrainer(args, cfg, audio_paket=audio_paket)
    
    # Lade den vortrainierten Modell-Checkpoint
    logger.info(f"Lade Checkpoint von: {args.test_ckpt}")
    other_tools.load_checkpoints(trainer.model, args.test_ckpt, args.g_name)
    
    # Starte den Inferenzprozess und erhalte den Pfad zur NPZ-Datei
    npz_datei_pfad = trainer.starte_inferenz_und_speichere_npz()
    
    return npz_datei_pfad

# --- Hauptausführungsblock ---
if __name__ == "__main__":
    # Setzen der Umgebungsvariablen, die vom Skript benötigt werden
    os.environ["MASTER_ADDR"] = '127.0.0.3'
    os.environ["MASTER_PORT"] = '8678'
    
    # 1. Kommandozeilenargumente parsen
    parser = argparse.ArgumentParser(description="Erzeuge 3D-Bewegungen aus einer Audiodatei mit einer Konfigurationsdatei.")
    parser.add_argument("--config", type=str, required=True, help="Pfad zur config.yaml Konfigurationsdatei.")
    parser.add_argument("--audio", type=str, required=True, help="Pfad zur Eingabe-Audiodatei (WAV).")
    cli_args = parser.parse_args()

    # 2. Konfiguration aus der YAML-Datei laden
    args, cfg = load_config_from_yaml(cli_args.config)
    
    # Audio-Pfad aus der Kommandozeile übernehmen
    audio_datei = cli_args.audio
    
    if not os.path.exists(audio_datei):
        logger.error(f"Audiodatei nicht gefunden unter: {audio_datei}")
        sys.exit(1)

    logger.info(f"Verarbeite Audiodatei: {audio_datei}")
    
    # 3. Rufe die Hauptfunktion auf, um die Bewegung zu erzeugen
    output_npz_pfad = erzeuge_bewegung_aus_audio(args, cfg, audio_datei)
    
    if output_npz_pfad:
        logger.info(f"Bewegungsdatei erfolgreich erstellt unter: {output_npz_pfad}")
    else:
        logger.error("Fehler bei der Erstellung der Bewegungsdatei.")