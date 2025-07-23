import os
import torch
from tqdm import tqdm
from loguru import logger

from dataloaders.data_tools import GestureTestData
from models.builder import build_model

class BaseTrainer:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        # Setup device
        self.device = torch.device(args.device)
        torch.set_grad_enabled(False)

        # Build model
        self.model = build_model(cfg, args).to(self.device)
        self.model.eval()

        # Prepare test data (TextGrid + Audio)
        self.test_loader = self._build_dataloader()

    def _build_dataloader(self):
        dataset = GestureTestData(
            audio_file_path=self.args.audio_file_path,
            textgrid_file_path=self.args.textgrid_file_path,
            gender=self.args.gender,
            fps=self.cfg.sample.fps
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        return loader

    def test_demo(self, epoch=999):
        logger.info(f"[Test] Starting inference (epoch={epoch})")

        for batch in tqdm(self.test_loader, desc="Running GestureLSM"):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)

            # Prediction
            result = self.model.infer(batch)
            return result  # Nur einen Clip verarbeiten