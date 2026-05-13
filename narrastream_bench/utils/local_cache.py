"""Local disk caches for resumable metric evaluation."""

from __future__ import annotations

from hashlib import sha1
import json
import os
from pathlib import Path

from narrastream_bench.utils.runtime_dependencies import NARRASTREAM_BENCH_ROOT, resolve_repo_path


def resolve_named_cache_dir(config=None, section=None, default_subdir=None) -> Path:
    cache_cfg = ((config or {}).get("cache") or {}).get(section or "", {})
    configured = resolve_repo_path(cache_cfg.get("cache_dir"))
    if configured is not None:
        return Path(configured)
    return NARRASTREAM_BENCH_ROOT / "cache" / str(default_subdir or section or "local_cache")


def _metric_cache_enabled(config=None) -> bool:
    cache_cfg = ((config or {}).get("cache") or {}).get("metric_results", {})
    return bool(cache_cfg.get("enabled", True))


def _metric_cache_root(run_output_path=None, config=None) -> Path | None:
    if not _metric_cache_enabled(config=config):
        return None

    cache_cfg = ((config or {}).get("cache") or {}).get("metric_results", {})
    configured = resolve_repo_path(cache_cfg.get("root_dir"))
    if configured is not None:
        if run_output_path:
            run_key = sha1(
                os.path.abspath(str(run_output_path)).encode("utf-8")
            ).hexdigest()[:12]
            return Path(configured) / run_key
        return Path(configured)

    if run_output_path:
        return Path(run_output_path) / "local_cache" / "metrics"

    return None


def _safe_token(value) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return safe[:80] or sha1(text.encode("utf-8")).hexdigest()[:16]


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


class MetricResultCache:
    def __init__(self, metric_name, *, run_output_path=None, config=None):
        self.metric_name = metric_name
        self.root_dir = _metric_cache_root(
            run_output_path=run_output_path,
            config=config,
        )
        metric_cfg = (
            (((config or {}).get("aggregation") or {}).get("metrics") or {}).get(
                metric_name,
                {},
            )
        )
        cache_version = metric_cfg.get("cache_version")
        metric_dir_name = metric_name
        if cache_version:
            metric_dir_name = f"{metric_name}__{_safe_token(cache_version)}"
        self.metric_dir = (
            self.root_dir / metric_dir_name if self.root_dir is not None else None
        )

    @property
    def enabled(self) -> bool:
        return self.metric_dir is not None

    def _sample_path(self, sample_id) -> Path | None:
        if not self.enabled:
            return None
        return self.metric_dir / "samples" / f"sample_{_safe_token(sample_id)}.json"

    def _item_path(self, namespace, sample_id, item_id) -> Path | None:
        if not self.enabled:
            return None
        return (
            self.metric_dir
            / "items"
            / str(namespace)
            / f"sample_{_safe_token(sample_id)}__item_{_safe_token(item_id)}.json"
        )

    def load_sample_state(self, sample_id):
        path = self._sample_path(sample_id)
        if path is None or not path.exists():
            return None
        return _read_json(path)

    def load_sample_score(self, sample_id):
        payload = self.load_sample_state(sample_id)
        if not isinstance(payload, dict) or "score" not in payload:
            return None
        return payload["score"]

    def save_sample_state(self, sample_id, payload) -> None:
        path = self._sample_path(sample_id)
        if path is None:
            return
        wrapped = {"sample_id": sample_id}
        if isinstance(payload, dict):
            wrapped.update(payload)
        else:
            wrapped["value"] = payload
        _write_json_atomic(path, wrapped)

    def save_sample_score(self, sample_id, score, **extra) -> None:
        payload = {"score": score}
        payload.update(extra)
        self.save_sample_state(sample_id, payload)

    def load_item_state(self, namespace, sample_id, item_id):
        path = self._item_path(namespace, sample_id, item_id)
        if path is None or not path.exists():
            return None
        return _read_json(path)

    def load_item_score(self, namespace, sample_id, item_id):
        payload = self.load_item_state(namespace, sample_id, item_id)
        if not isinstance(payload, dict) or "score" not in payload:
            return None
        return payload["score"]

    def save_item_state(self, namespace, sample_id, item_id, payload) -> None:
        path = self._item_path(namespace, sample_id, item_id)
        if path is None:
            return
        wrapped = {
            "sample_id": sample_id,
            "item_id": item_id,
        }
        if isinstance(payload, dict):
            wrapped.update(payload)
        else:
            wrapped["value"] = payload
        _write_json_atomic(path, wrapped)

    def save_item_score(self, namespace, sample_id, item_id, score, **extra) -> None:
        payload = {"score": score}
        payload.update(extra)
        self.save_item_state(namespace, sample_id, item_id, payload)


def build_metric_result_cache(metric_name, *, run_output_path=None, config=None):
    return MetricResultCache(
        metric_name,
        run_output_path=run_output_path,
        config=config,
    )


def split_cached_samples(eval_data, metric_cache: MetricResultCache):
    cached_results = {}
    pending_samples = []

    for sample in eval_data:
        sample_state = metric_cache.load_sample_state(sample["sample_id"])
        if not isinstance(sample_state, dict) or "score" not in sample_state:
            pending_samples.append(sample)
            continue
        cached_results[sample["sample_id"]] = sample_state.get("score")

    return cached_results, pending_samples
