"""DINO 编码器封装"""
import torch
import torch.nn.functional as F

from narrastream_bench.third_party.dino_vit import (
    build_dino_model,
    resolve_dino_feature_dim,
    resolve_dino_weights_url,
)
from narrastream_bench.utils.runtime_dependencies import load_torch_state_dict, resolve_repo_path
from narrastream_bench.utils.video_primitives import build_dino_transform


class DINOEncoder:
    def __init__(self, device='cuda', config=None, path_config=None):
        self.device = device

        # 从配置读取参数
        model_config = config.get('models', {}).get('dino', {}) if config else {}
        paths = path_config.get('dino', {}) if path_config else {}

        model_name = model_config.get('model_name', 'dino_vitb16')
        input_size = model_config.get('input_size', 224)
        weights_path = resolve_repo_path(paths.get('weights_path'))

        self.feature_dim = resolve_dino_feature_dim(model_name)
        self.model = build_dino_model(model_name)
        if weights_path:
            state_dict = load_torch_state_dict(weights_path)
        else:
            state_dict = torch.hub.load_state_dict_from_url(
                resolve_dino_weights_url(model_name),
                map_location='cpu',
            )
        self.model.load_state_dict(state_dict, strict=True)

        self.model = self.model.to(device)
        self.model.eval()
        self.transform = build_dino_transform(input_size)

    def encode_frame(self, frame):
        """编码单帧 (PIL Image)"""
        with torch.no_grad():
            tensor = self.transform(frame).unsqueeze(0).to(self.device)
            feat = self.model(tensor)
            return F.normalize(feat, dim=-1)

    def encode_segment(self, frames):
        """编码视频段 (list of PIL)，返回平均特征"""
        if len(frames) == 0:
            return torch.zeros(1, self.feature_dim, device=self.device)

        n = len(frames)
        indices = sorted(set([0, n//4, n//2, 3*n//4, n-1]))
        indices = [i for i in indices if i < n]

        feats = [self.encode_frame(frames[i]) for i in indices]
        return torch.stack(feats).mean(dim=0)
