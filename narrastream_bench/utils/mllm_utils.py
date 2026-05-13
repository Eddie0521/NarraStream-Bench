"""Cached MLLM helpers for API-backed metrics."""

from __future__ import annotations

from hashlib import sha1
import json
from pathlib import Path

from narrastream_bench.utils.api_clients import build_openai_compatible_client, resolve_service_config
from narrastream_bench.utils.api_retry import resolve_retry_policy, run_with_retry
from narrastream_bench.utils.runtime_dependencies import NARRASTREAM_BENCH_ROOT, resolve_repo_path


SCENE_CHANGE_CACHE_VERSION = "v2_retry"
ENTITY_GROUP_CACHE_VERSION = "v2_retry"
_MEMORY_CACHE: dict[str, object] = {}


def _metric_cache_dir(config=None, metric_name=None, default_subdir=None) -> Path:
    metric_cfg = ((config or {}).get("aggregation") or {}).get("metrics", {})
    per_metric_cfg = metric_cfg.get(metric_name or "", {})
    cache_dir = per_metric_cfg.get("cache_dir")
    resolved = resolve_repo_path(cache_dir)
    if resolved is not None:
        return Path(resolved)
    return NARRASTREAM_BENCH_ROOT / "cache" / default_subdir


def _service_config(config=None):
    return resolve_service_config(config=config, service_name="mllm")


def _response_text(response) -> str:
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


def _extract_json(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def _cache_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"{cache_key}.json"


def _load_cache(cache_path: Path):
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_cache(cache_path: Path, payload) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _cached_json_completion(
    *,
    cache_namespace: str,
    cache_dir: Path,
    cache_payload,
    prompt: str,
    config=None,
    request_label=None,
):
    service_config = _service_config(config=config)
    cache_key = sha1(
        json.dumps(
            {
                "namespace": cache_namespace,
                "service_model": service_config.model,
                "payload": cache_payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    memory_key = f"{cache_namespace}:{cache_key}"
    if memory_key in _MEMORY_CACHE:
        return _MEMORY_CACHE[memory_key]

    path = _cache_path(cache_dir, cache_key)
    cached_payload = _load_cache(path)
    if cached_payload is not None:
        _MEMORY_CACHE[memory_key] = cached_payload
        return cached_payload

    client = build_openai_compatible_client(service_config)
    retry_policy = resolve_retry_policy(
        config=config,
        service_name="mllm",
        default_max_attempts=10,
        default_initial_delay=2.0,
        default_max_delay=45.0,
        default_backoff=1.8,
    )
    label = request_label or f"mllm_json cache={cache_key[:8]}"

    def _request_and_parse():
        response = client.chat.completions.create(
            model=service_config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=service_config.max_tokens,
            temperature=service_config.temperature,
            response_format={"type": "json_object"},
        )
        return _extract_json(_response_text(response))

    parsed = run_with_retry(
        _request_and_parse,
        label=label,
        policy=retry_policy,
        retryable=lambda exc: True,
    )
    _save_cache(path, parsed)
    _MEMORY_CACHE[memory_key] = parsed
    return parsed


def batch_judge_scene_changes(prompts, config=None):
    """Judge all adjacent prompt transitions with a single cached JSON call."""
    prompts = list(prompts or [])
    if len(prompts) < 2:
        return []

    prompt_lines = "\n".join(f"{idx}. {prompt}" for idx, prompt in enumerate(prompts))
    prompt = f"""分析以下视频片段描述序列中相邻片段之间，哪些应该保持视觉连续性（同一场景、同一人物或明显延续）。

输入片段：
{prompt_lines}

请对每一对相邻片段给出布尔判断。长度必须是 {len(prompts) - 1}。

输出格式：
{{"keep":[true,false,true]}}
"""

    payload = _cached_json_completion(
        cache_namespace=f"scene_change:{SCENE_CHANGE_CACHE_VERSION}",
        cache_dir=_metric_cache_dir(
            config=config,
            metric_name="conditional_adjacent",
            default_subdir="conditional_adjacent",
        ),
        cache_payload={"prompts": prompts},
        prompt=prompt,
        config=config,
        request_label=f"scene_change segments={len(prompts)}",
    )
    keep = payload.get("keep")
    if not isinstance(keep, list) or len(keep) != len(prompts) - 1:
        raise ValueError(f"Invalid keep payload: {payload}")
    return [bool(item) for item in keep]


def mllm_judge_scene_change(prompt_i, prompt_i1, config=None):
    """Compatibility wrapper for judging one adjacent transition."""
    keep_flags = batch_judge_scene_changes([prompt_i, prompt_i1], config=config)
    return bool(keep_flags[0]) if keep_flags else False


def extract_entity_groups_cached(prompts, config=None):
    """Extract entity groups once per prompt sequence with a cached JSON call."""
    prompts = list(prompts or [])
    if not prompts:
        return {}

    prompt_text = "\n".join([f"{i}: {p}" for i, p in enumerate(prompts)])
    prompt = f"""分析以下视频片段描述序列，识别出现的主要实体，并标注每个实体在哪些片段中出现（0-indexed）。

{prompt_text}

只输出 JSON，例如：
{{"wizard":[0,1,2,3], "dragon":[1,2,3,4]}}
"""

    payload = _cached_json_completion(
        cache_namespace=f"entity_groups:{ENTITY_GROUP_CACHE_VERSION}",
        cache_dir=_metric_cache_dir(
            config=config,
            metric_name="conditional_longrange",
            default_subdir="conditional_longrange",
        ),
        cache_payload={"prompts": prompts},
        prompt=prompt,
        config=config,
        request_label=f"entity_groups segments={len(prompts)}",
    )
    if not isinstance(payload, dict):
        return {"main_subject": list(range(len(prompts)))}

    normalized = {}
    for key, indices in payload.items():
        if not isinstance(indices, list):
            continue
        cleaned = []
        for index in indices:
            try:
                int_index = int(index)
            except (TypeError, ValueError):
                continue
            if 0 <= int_index < len(prompts):
                cleaned.append(int_index)
        if cleaned:
            normalized[str(key)] = cleaned

    return normalized or {"main_subject": list(range(len(prompts)))}


def mllm_extract_entity_groups(prompts, config=None):
    """Compatibility wrapper for cached entity-group extraction."""
    return extract_entity_groups_cached(prompts, config=config)
