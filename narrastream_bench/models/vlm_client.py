"""VLM client with local frame caching."""

from __future__ import annotations

import base64
from hashlib import sha1
import json
import os
from pathlib import Path
from threading import Lock, get_ident
from typing import Iterable

from narrastream_bench.utils.api_clients import build_openai_compatible_client, resolve_service_config
from narrastream_bench.utils.local_cache import resolve_named_cache_dir


_FRAME_CACHE: dict[tuple[str, int, int, int], list[str]] = {}
_FRAME_CACHE_HITS = 0
_FRAME_CACHE_MISSES = 0
_FRAME_CACHE_LOCK = Lock()


def _video_cache_key(video_path: str, num_frames: int) -> tuple[str, int, int, int]:
    stat = os.stat(video_path)
    return (
        os.path.abspath(video_path),
        int(num_frames),
        int(stat.st_mtime_ns),
        int(stat.st_size),
    )


def _frame_cache_file(cache_key) -> str:
    path, num_frames, mtime_ns, file_size = cache_key
    payload = json.dumps(
        {
            "path": path,
            "num_frames": num_frames,
            "mtime_ns": mtime_ns,
            "file_size": file_size,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return f"{sha1(payload.encode('utf-8')).hexdigest()}.json"


class VLMClient:
    def __init__(self, model=None, config=None, path_config=None):
        del path_config
        service_config = resolve_service_config(
            config=config,
            service_name="vlm",
            model_override=model,
        )
        self.model = service_config.model
        self.max_tokens = service_config.max_tokens
        self.temperature = service_config.temperature
        self.frames_per_segment = (
            (config or {}).get("evaluation", {}).get("vlm_frames_per_segment", 5)
        )
        frame_cache_cfg = ((config or {}).get("cache") or {}).get("vlm_frames", {})
        self.frame_cache_enabled = bool(frame_cache_cfg.get("enabled", True))
        self.frame_cache_dir = resolve_named_cache_dir(
            config=config,
            section="vlm_frames",
            default_subdir="vlm_frames",
        )
        self.client = build_openai_compatible_client(service_config)

    @staticmethod
    def _sample_indices(total_frames, num_frames):
        if total_frames <= 0 or num_frames <= 0:
            return []
        if total_frames <= num_frames:
            return list(range(total_frames))
        if num_frames == 1:
            return [total_frames // 2]
        indices = []
        for i in range(num_frames):
            position = round(i * (total_frames - 1) / (num_frames - 1))
            indices.append(position)
        return sorted(set(indices))

    def _encode_segment_frames(self, video_path, num_frames=5):
        import cv2

        global _FRAME_CACHE_HITS, _FRAME_CACHE_MISSES

        cache_key = _video_cache_key(video_path, num_frames)
        with _FRAME_CACHE_LOCK:
            cached = _FRAME_CACHE.get(cache_key)
        if cached is not None:
            with _FRAME_CACHE_LOCK:
                _FRAME_CACHE_HITS += 1
            return cached

        cache_path = None
        if self.frame_cache_enabled:
            cache_path = Path(self.frame_cache_dir) / _frame_cache_file(cache_key)
            if cache_path.exists():
                try:
                    cached_frames = json.loads(cache_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    cached_frames = None
                if isinstance(cached_frames, list) and all(
                    isinstance(item, str) for item in cached_frames
                ):
                    with _FRAME_CACHE_LOCK:
                        _FRAME_CACHE[cache_key] = cached_frames
                        _FRAME_CACHE_HITS += 1
                    return cached_frames

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            with _FRAME_CACHE_LOCK:
                _FRAME_CACHE_MISSES += 1
                _FRAME_CACHE[cache_key] = []
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = cache_path.with_suffix(
                    f"{cache_path.suffix}.tmp.{os.getpid()}.{get_ident()}"
                )
                tmp_path.write_text("[]", encoding="utf-8")
                os.replace(tmp_path, cache_path)
            return []

        encoded_frames = []
        for frame_index in self._sample_indices(total_frames, num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue
            ok, buffer = cv2.imencode(".jpg", frame)
            if ok:
                encoded_frames.append(base64.b64encode(buffer).decode("utf-8"))

        cap.release()
        with _FRAME_CACHE_LOCK:
            _FRAME_CACHE_MISSES += 1
            _FRAME_CACHE[cache_key] = encoded_frames
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(
                f"{cache_path.suffix}.tmp.{os.getpid()}.{get_ident()}"
            )
            tmp_path.write_text(
                json.dumps(encoded_frames, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp_path, cache_path)
        return encoded_frames

    def _build_segment_content(self, segment_paths: Iterable[str], prompt: str, frames_per_segment: int):
        content = [{"type": "text", "text": prompt}]
        for segment_idx, segment_path in enumerate(segment_paths, start=1):
            content.append(
                {
                    "type": "text",
                    "text": f"以下是第 {segment_idx} 段视频的关键帧（按时间顺序采样）。",
                }
            )
            for frame_b64 in self._encode_segment_frames(
                segment_path,
                num_frames=frames_per_segment,
            ):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_b64}",
                        },
                    }
                )
        return content

    @staticmethod
    def _response_text(response):
        message = response.choices[0].message
        content = getattr(message, "content", "")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(str(text))
                else:
                    text = getattr(item, "text", None) or getattr(item, "content", None)
                    if text:
                        parts.append(str(text))
            if parts:
                return "\n".join(parts).strip()

        reasoning = getattr(message, "reasoning_content", None)
        if isinstance(reasoning, str):
            return reasoning.strip()

        return ""

    def evaluate_segments(
        self,
        segment_paths,
        prompt,
        frames_per_segment=None,
        response_format=None,
        extra_body=None,
        max_tokens=None,
    ):
        frames_per_segment = frames_per_segment or self.frames_per_segment
        content = self._build_segment_content(segment_paths, prompt, frames_per_segment)
        request_kwargs = dict(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=int(max_tokens or self.max_tokens),
            temperature=self.temperature,
        )
        if response_format is not None:
            request_kwargs["response_format"] = response_format
        if extra_body is not None:
            request_kwargs["extra_body"] = extra_body
        response = self.client.chat.completions.create(**request_kwargs)
        return self._response_text(response)

    @staticmethod
    def cache_stats():
        with _FRAME_CACHE_LOCK:
            return {
                "entries": len(_FRAME_CACHE),
                "hits": _FRAME_CACHE_HITS,
                "misses": _FRAME_CACHE_MISSES,
            }
