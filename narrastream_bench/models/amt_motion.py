"""AMT-S runtime wrapper used as a NarraStream-Bench motion backbone."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from narrastream_bench.utils.runtime_dependencies import resolve_repo_path


NARRASTREAM_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(NARRASTREAM_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(NARRASTREAM_BENCH_ROOT))

from narrastream_bench.third_party.amt.utils.build_utils import build_from_cfg
from narrastream_bench.third_party.amt.utils.utils import (
    InputPadder,
    img2tensor,
    tensor2img,
)


class _FrameProcess:
    def get_frames(self, video_path: str):
        frame_list = []
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame)
        video.release()
        if not frame_list:
            raise ValueError(f"No frames extracted from video: {video_path}")
        return frame_list

    @staticmethod
    def extract_frame(frame_list, start_from=0):
        return [frame_list[i] for i in range(start_from, len(frame_list), 2)]


class AMTMotionSmoothness:
    def __init__(self, device="cuda", config=None, path_config=None):
        self.device = device
        self.niters = 1
        self.fp = _FrameProcess()
        metric_cfg = ((config or {}).get("aggregation") or {}).get("metrics", {})
        motion_cfg = metric_cfg.get("motion_smoothness", {})
        self.score_mapping = str(motion_cfg.get("score_mapping", "exp"))
        self.score_tau = float(motion_cfg.get("tau", 3.5))
        if self.score_tau <= 0:
            raise ValueError(
                f"motion_smoothness tau must be positive, got {self.score_tau}"
            )

        amt_paths = (path_config or {}).get("amt", {})
        config_path = resolve_repo_path(amt_paths.get("config_path"))
        checkpoint_path = resolve_repo_path(amt_paths.get("checkpoint_path"))

        default_dir = NARRASTREAM_BENCH_ROOT / "pretrained" / "amt_model"
        self.config_path = config_path or (default_dir / "AMT-S.yaml")
        self.checkpoint_path = checkpoint_path or (default_dir / "amt-s.pth")

        if not self.config_path.exists():
            raise FileNotFoundError(f"AMT config file not found: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"AMT checkpoint not found: {self.checkpoint_path}"
            )

        self._initialization()
        self._load_model()

    def _load_model(self):
        with self.config_path.open("r", encoding="utf-8") as file:
            network_cfg = yaml.safe_load(file)["network"]

        self.model = build_from_cfg(network_cfg)
        ckpt = torch.load(
            str(self.checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

    def _initialization(self):
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            self.anchor_resolution = 1024 * 512
            self.anchor_memory = 1500 * 1024**2
            self.anchor_memory_bias = 2500 * 1024**2
            self.vram_total = torch.cuda.get_device_properties(self.device).total_memory
        else:
            self.anchor_resolution = 8192 * 8192
            self.anchor_memory = 1
            self.anchor_memory_bias = 0
            self.vram_total = 1

        self.embt = torch.tensor(0.5, dtype=torch.float32).view(1, 1, 1, 1).to(self.device)

    def _available_vram(self) -> int:
        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            return 1

        try:
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            return max(int(free_bytes), 1)
        except Exception:
            return max(int(self.vram_total), 1)

    @staticmethod
    def _frame_tensor(frame, desired_shape):
        tensor = img2tensor(frame)
        if tensor.shape[-2:] != desired_shape:
            tensor = F.interpolate(tensor, size=tuple(desired_shape), mode="bilinear")
        return tensor

    def evaluate_video(self, video_path: str) -> dict[str, float | str]:
        frames = self.fp.get_frames(video_path)
        frame_list = self.fp.extract_frame(frames, start_from=0)
        if len(frame_list) <= 1:
            return {
                "raw_error": 0.0,
                "score": 1.0,
                "score_mapping": self.score_mapping,
                "tau": self.score_tau,
            }

        desired_shape = frame_list[0].shape[:2]
        if any(frame.shape[:2] != desired_shape for frame in frame_list[1:]):
            print(
                "Inconsistent size of input video frames. "
                f"All frames will be resized to {desired_shape}"
            )

        prev_input = self._frame_tensor(frame_list[0], desired_shape)
        h, w = prev_input.shape[-2:]
        vram_avail = self._available_vram()
        scale = self.anchor_resolution / (h * w) * np.sqrt(
            max(vram_avail - self.anchor_memory_bias, 1) / self.anchor_memory
        )
        scale = 1 if scale > 1 else scale
        scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
        padding = int(16 / scale)
        padder = InputPadder(prev_input.shape, padding)

        outputs = [tensor2img(prev_input)]
        for frame in frame_list[1:]:
            curr_input = self._frame_tensor(frame, desired_shape)
            prev_padded = padder.pad(prev_input)
            curr_padded = padder.pad(curr_input)
            with torch.inference_mode():
                pred = self.model(
                    prev_padded.to(self.device),
                    curr_padded.to(self.device),
                    self.embt,
                    scale_factor=scale,
                    eval=True,
                )["imgt_pred"]
            outputs.append(tensor2img(padder.unpad(pred.cpu())))
            outputs.append(tensor2img(curr_input))
            prev_input = curr_input
            del prev_padded, curr_padded, pred

        raw_error = self._vfi_score(frames, outputs)
        return {
            "raw_error": raw_error,
            "score": self._score_from_error(raw_error),
            "score_mapping": self.score_mapping,
            "tau": self.score_tau,
        }

    def score_video(self, video_path: str) -> float:
        return float(self.evaluate_video(video_path)["score"])

    def _score_from_error(self, raw_error: float) -> float:
        if self.score_mapping != "exp":
            raise ValueError(
                f"Unsupported motion_smoothness score_mapping: {self.score_mapping}"
            )
        return float(math.exp(-float(raw_error) / self.score_tau))

    def _vfi_score(self, ori_frames, interpolate_frames):
        ori = self.fp.extract_frame(ori_frames, start_from=1)
        interpolate = self.fp.extract_frame(interpolate_frames, start_from=1)
        scores = [
            float(np.mean(cv2.absdiff(img1, img2)))
            for img1, img2 in zip(ori, interpolate)
        ]
        return float(np.mean(np.array(scores))) if scores else 0.0
