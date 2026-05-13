"""LanguageBind 编码器封装"""
from hashlib import sha1
import os
from pathlib import Path
import sys

import torch

VENDORED_LANGUAGEBIND_SRC = Path(__file__).resolve().parents[1] / "third_party" / "languagebind" / "src"
if str(VENDORED_LANGUAGEBIND_SRC) not in sys.path:
    sys.path.insert(0, str(VENDORED_LANGUAGEBIND_SRC))

from languagebind import (
    LanguageBind,
    LanguageBindVideoTokenizer,
    to_device,
    transform_dict,
)

from narrastream_bench.utils.local_cache import resolve_named_cache_dir
from narrastream_bench.utils.runtime_dependencies import resolve_repo_path


class LanguageBindEncoder:
    def __init__(self, device='cuda', config=None, path_config=None):
        self.device = device
        
        # 从配置读取参数
        model_config = config.get('models', {}).get('languagebind', {}) if config else {}
        paths = path_config.get('languagebind', {}) if path_config else {}
        
        raw_cache_dir = paths.get('cache_dir') or model_config.get('cache_dir', './pretrained')
        cache_dir = resolve_repo_path(raw_cache_dir)
        raw_video_model = paths.get('video_model') or model_config.get('video_model', 'LanguageBind_Video_FT')
        resolved_video_model = resolve_repo_path(raw_video_model)
        if resolved_video_model and resolved_video_model.exists():
            video_model = str(resolved_video_model)
        else:
            video_model = raw_video_model

        pretrained_ckpt = (
            video_model
            if Path(video_model).expanduser().exists() or "/" in video_model
            else f"LanguageBind/{video_model}"
        )
        self.pretrained_ckpt = pretrained_ckpt
        languagebind_cache_cfg = ((config or {}).get("cache") or {}).get(
            "languagebind",
            {},
        )
        self.embedding_cache_enabled = bool(languagebind_cache_cfg.get("enabled", True))
        self.embedding_cache_dir = resolve_named_cache_dir(
            config=config,
            section="languagebind",
            default_subdir="languagebind",
        )
        self._memory_cache = {}
        
        # 加载 Video + Language 模型
        self.model = LanguageBind(
            clip_type={
                'video': pretrained_ckpt,
            },
            cache_dir=str(cache_dir)
        )
        self.model = self.model.to(device)
        self.model.eval()

        # LanguageBind 仅提供模态 processor；文本侧直接复用视频分支 tokenizer。
        self.tokenizer = LanguageBindVideoTokenizer.from_pretrained(
            pretrained_ckpt,
            cache_dir=str(cache_dir),
        )
        self.video_transform = transform_dict['video'](
            self.model.modality_config['video'],
            self.tokenizer,
        )

    def encode_video(self, video_path):
        """编码视频文件 (路径)"""
        cache_key = self._video_cache_key(video_path)
        cached = self._load_cached_feature(cache_key)
        if cached is not None:
            return cached

        with torch.no_grad():
            video_inputs = self.video_transform(
                images=[video_path],
                return_tensors='pt',
            )
            video_inputs = to_device(video_inputs, self.device)
            feat = self.model({'video': video_inputs})['video']
            feature = feat.squeeze(0)
            self._save_cached_feature(cache_key, feature)
            return feature

    def encode_text(self, text):
        """编码文本"""
        cache_key = self._text_cache_key(text)
        cached = self._load_cached_feature(cache_key)
        if cached is not None:
            return cached

        with torch.no_grad():
            language_inputs = self.tokenizer(
                [text],
                max_length=77,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            language_inputs = to_device(language_inputs, self.device)
            feat = self.model({'language': language_inputs})['language']
            feature = feat.squeeze(0)
            self._save_cached_feature(cache_key, feature)
            return feature

    def _cache_path(self, cache_key):
        return Path(self.embedding_cache_dir) / f"{cache_key}.pt"

    def _load_cached_feature(self, cache_key):
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key].to(self.device)

        if not self.embedding_cache_enabled:
            return None

        path = self._cache_path(cache_key)
        if not path.exists():
            return None

        try:
            cached = torch.load(str(path), map_location="cpu")
        except Exception:
            return None

        if not isinstance(cached, torch.Tensor):
            return None

        cached = cached.detach().float()
        self._memory_cache[cache_key] = cached
        return cached.to(self.device)

    def _save_cached_feature(self, cache_key, tensor):
        cached = tensor.detach().float().cpu()
        self._memory_cache[cache_key] = cached

        if not self.embedding_cache_enabled:
            return

        path = self._cache_path(cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
        torch.save(cached, str(tmp_path))
        os.replace(tmp_path, path)

    def _video_cache_key(self, video_path):
        stat = Path(video_path).stat()
        payload = f"video::{self.pretrained_ckpt}::{Path(video_path).resolve()}::{stat.st_mtime_ns}::{stat.st_size}"
        return sha1(payload.encode("utf-8")).hexdigest()

    def _text_cache_key(self, text):
        payload = f"text::{self.pretrained_ckpt}::{text}"
        return sha1(payload.encode("utf-8")).hexdigest()
