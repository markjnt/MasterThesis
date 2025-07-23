import os
import sys
import torch
import warnings
import numpy as np
from loguru import logger

from utils import other_tools, other_tools_hf
from base_trainer_simplified import BaseTrainer  # muss angepasst sein – ohne Whisper, ohne MFA

@logger.catch
def run_gesturelsm(audio_path, text_path, textgrid_path, output_path):
    class Args:
        # Modellparameter
        test_ckpt = "checkpoints/gesturelsm/model.ckpt"
        g_name = "GestureLSM"

        # Input-Pfade
        audio_file_path = audio_path
        textgrid_file_path = textgrid_path
        tmp_dir = os.path.dirname(output_path)

        # Hardware/Logging
        seed = 42
        debug = False
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Sample settings (werden ggf. im Trainer/Model verwendet)
        fps = 30
        gender = "neutral"

    args = Args()

    # Dummy-CFG: du kannst bei Bedarf noch mehr Attribute hinzufügen
    class Cfg:
        class Model:
            name = "GestureLSM"
            vq_type = "rvq"  # oder "vq", je nachdem

        class Sample:
            fps = 30
            num_steps = 100
            ddim_eta = 0.0

        model = Model()
        sample = Sample()

    cfg = Cfg()

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Text not found: {text_path}")
    if not os.path.exists(textgrid_path):
        raise FileNotFoundError(f"TextGrid not found: {textgrid_path}")

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    other_tools_hf.set_random_seed(args)
    other_tools_hf.print_exp_info(args)

    # Modell & Trainer vorbereiten
    trainer = BaseTrainer(args, cfg)
    other_tools.load_checkpoints(trainer.model, args.test_ckpt, args.g_name)

    result = trainer.test_demo(epoch=999)

    # Speichern als .npz
    npz_path = os.path.join(args.tmp_dir, "gesture_output.npz")
    np.savez(npz_path,
             poses=result['rec_pose'].cpu().numpy(),
             trans=result['rec_trans'].cpu().numpy(),
             betas=result['tar_beta'].cpu().numpy(),
             expressions=result['rec_exps'].cpu().numpy(),
             model='smplx2020',
             gender=args.gender,
             mocap_frame_rate=cfg.sample.fps)

    logger.info(f"Saved motion to {npz_path}")
    return npz_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to input audio .wav")
    parser.add_argument("--text", required=True, help="Path to input text .txt (not used)")
    parser.add_argument("--textgrid", required=True, help="Path to TextGrid file")
    parser.add_argument("--out", default="output/gesture_output.npz", help="Path to save output .npz")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    run_gesturelsm(args.audio, args.text, args.textgrid, args.out)