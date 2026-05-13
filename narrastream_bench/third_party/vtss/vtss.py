"""Vendored VTSS calculator adapted from IVEBench for NarraStream-Bench."""
from pathlib import Path
import logging
import time

import cv2
import numpy as np
import torch
import yaml

from .training_suitability_assessment.datasets import FusionDataset
from .training_suitability_assessment.model import DiViDeAddEvaluator


LOGGER = logging.getLogger(__name__)
DEFAULT_INFER_CONFIG = (
    Path(__file__).resolve().parent / "training_suitability_assessment" / "infer.yml"
)


def _release_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_config_path(config_path):
    if config_path is None:
        return DEFAULT_INFER_CONFIG
    return Path(config_path).expanduser().resolve()


class VTSSCalculator:
    def __init__(self, device, config_path=None, checkpoint_path=None):
        self.device = device
        self.config_path = _resolve_config_path(config_path)
        self.checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path
            else None
        )

        if not self.config_path.exists():
            raise FileNotFoundError(f"VTSS config file not found: {self.config_path}")
        if self.checkpoint_path is None:
            raise ValueError(
                "VTSS checkpoint_path is required. Set vtss.checkpoint_path in configs/paths.yaml."
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"VTSS model weights not found: {self.checkpoint_path}")

        self._load_model()

    def _load_model(self):
        with self.config_path.open("r", encoding="utf-8") as file:
            opt = yaml.safe_load(file)

        self.model = DiViDeAddEvaluator(**opt["model"]["args"]).to(self.device)
        self.model.eval()

        state_dict = torch.load(
            str(self.checkpoint_path),
            map_location=self.device,
            weights_only=False,
        )["state_dict"]
        self.model.load_state_dict(state_dict, strict=True)

        self.val_dataset = FusionDataset(opt["data"]["test-data"]["args"])

    def process_video_from_frames(self, frame_folder_path):
        frame_dir = Path(frame_folder_path)
        if not frame_dir.exists():
            raise FileNotFoundError(f"Frame folder not found: {frame_dir}")

        frame_files = sorted(
            [
                frame_path
                for frame_path in frame_dir.iterdir()
                if frame_path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]
        )
        if not frame_files:
            raise ValueError(f"No image files found in {frame_dir}")

        temp_video_path = self._create_temp_video_from_frames(frame_files)
        try:
            return self.process_video(temp_video_path)
        finally:
            temp_video_path.unlink(missing_ok=True)

    def _create_temp_video_from_frames(self, frame_files):
        temp_video_path = frame_files[0].parent / "temp_vtss_video.mp4"
        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {frame_files[0]}")

        height, width, _ = first_frame.shape
        writer = cv2.VideoWriter(
            str(temp_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            24,
            (width, height),
        )

        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                writer.write(frame)
        writer.release()
        return temp_video_path

    def process_video(self, video_path):
        start_time = time.perf_counter()

        try:
            _release_cuda_cache()
            data = self.val_dataset.prepare_video(str(video_path))
            video = {}

            for key in ["resize", "fragments", "crop", "arp_resize", "arp_fragments"]:
                if key in data:
                    video[key] = data[key].to(self.device).unsqueeze(0)
                    b, c, t, h, w = video[key].shape
                    video[key] = video[key].reshape(
                        b,
                        c,
                        data["num_clips"][key],
                        t // data["num_clips"][key],
                        h,
                        w,
                    ).permute(0, 2, 1, 3, 4, 5).reshape(
                        b * data["num_clips"][key],
                        c,
                        t // data["num_clips"][key],
                        h,
                        w,
                    )

            with torch.no_grad():
                labels = self.model(video, reduce_scores=False)
                labels = [np.mean(label.cpu().numpy()) for label in labels]

            LOGGER.debug(
                "VTSS processing time %.2fs for %s",
                time.perf_counter() - start_time,
                video_path,
            )
            _release_cuda_cache()
            return float(labels[0])
        except Exception as exc:
            _release_cuda_cache()
            LOGGER.error("Error processing video %s: %s", video_path, exc)
            return None
