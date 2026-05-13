"""CLIP/ViCLIP 编码器封装"""
from pathlib import Path

import torch
import clip

from narrastream_bench.utils.runtime_dependencies import resolve_repo_path


class ViCLIPEncoder:
    def __init__(self, device='cuda', config=None, path_config=None):
        self.device = device
        
        # 从配置读取参数
        model_config = config.get('models', {}).get('clip', {}) if config else {}
        paths = path_config.get('clip', {}) if path_config else {}
        
        model_name = model_config.get('model_name', 'ViT-B/32')
        weights_path = resolve_repo_path(paths.get('weights_path'))
        
        # 加载模型
        if weights_path:
            if Path(weights_path).is_file() or Path(weights_path).suffix:
                self.model, self.preprocess = clip.load(str(weights_path), device=device)
            else:
                self.model, self.preprocess = clip.load(
                    model_name,
                    device=device,
                    download_root=str(weights_path),
                )
        else:
            self.model, self.preprocess = clip.load(model_name, device=device)
    
    def encode_text(self, text):
        """编码文本"""
        with torch.no_grad():
            tokens = clip.tokenize([text], truncate=True).to(self.device)
            return self.model.encode_text(tokens)
    
    def encode_video(self, frames):
        """编码视频段 (list of PIL)，返回平均特征"""
        if len(frames) == 0:
            return torch.zeros(1, 512).to(self.device)
        
        n = len(frames)
        indices = sorted(set([0, n//2, n-1]))
        indices = [i for i in indices if i < n]
        
        feats = []
        with torch.no_grad():
            for i in indices:
                img = self.preprocess(frames[i]).unsqueeze(0).to(self.device)
                feats.append(self.model.encode_image(img))
        
        return torch.stack(feats).mean(dim=0)
    
    def compute_similarity(self, frames, text):
        """计算视频-文本相似度"""
        video_feat = self.encode_video(frames)
        text_feat = self.encode_text(text)
        
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        
        return (video_feat @ text_feat.T).item()
