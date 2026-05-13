"""RAFT 光流模型封装"""
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from narrastream_bench.third_party.raft.core.raft import RAFT
from narrastream_bench.third_party.raft.core.utils_core.utils import InputPadder
from narrastream_bench.utils.runtime_dependencies import load_torch_state_dict, resolve_repo_path


class _RAFTArgs(SimpleNamespace):
    def __contains__(self, key):
        return hasattr(self, key)


class RAFTFlow:
    def __init__(self, device='cuda', config=None, path_config=None):
        self.device = device

        model_config = config.get('models', {}).get('raft', {}) if config else {}
        paths = path_config.get('raft', {}) if path_config else {}
        weights_path = resolve_repo_path(paths.get('weights_path'))

        if not weights_path:
            raise FileNotFoundError(
                "RAFT weights_path is required. "
                "Use a RAFT Things checkpoint such as 'raft-things.pth'."
            )

        self.iters = int(model_config.get('iters', 20))
        raft_args = _RAFTArgs(
            model=str(weights_path),
            small=bool(model_config.get('small', False)),
            mixed_precision=bool(model_config.get('mixed_precision', False)),
            alternate_corr=bool(model_config.get('alternate_corr', False)),
        )

        self.model = RAFT(raft_args)
        self.model.load_state_dict(load_torch_state_dict(weights_path), strict=True)
        self.model = self.model.to(device)
        self.model.eval()

    def compute_flow(self, frame1, frame2):
        """计算两帧之间的光流"""
        # 转换 PIL 到 tensor
        if isinstance(frame1, Image.Image):
            frame1 = torch.from_numpy(np.array(frame1)).permute(2, 0, 1).float()
        if isinstance(frame2, Image.Image):
            frame2 = torch.from_numpy(np.array(frame2)).permute(2, 0, 1).float()

        img1 = frame1.unsqueeze(0).to(self.device)
        img2 = frame2.unsqueeze(0).to(self.device)
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)

        with torch.no_grad():
            _, flow = self.model(img1, img2, iters=self.iters, test_mode=True)

        return padder.unpad(flow).squeeze(0)  # [2, H, W]
