"""LLM-based prompt importance planning for segment weighting."""
from __future__ import annotations

from hashlib import sha1
import json
from pathlib import Path

from narrastream_bench.utils.api_clients import build_openai_compatible_client, resolve_service_config
from narrastream_bench.utils.api_retry import resolve_retry_policy, run_with_retry
from narrastream_bench.utils.runtime_dependencies import NARRASTREAM_BENCH_ROOT, resolve_repo_path


PROMPT_WEIGHT_PLANNER_VERSION = "v4_mid_transition_focus_range100"
_MEMORY_CACHE: dict[str, list[float]] = {}


def _planner_config(config=None):
    aggregation_cfg = ((config or {}).get("aggregation") or {}).get(
        "llm_prompt_weighting", {}
    )
    service_name = aggregation_cfg.get("service_name", "planner")
    cache_dir = (
        resolve_repo_path(aggregation_cfg.get("cache_dir"))
        or (NARRASTREAM_BENCH_ROOT / "cache" / "prompt_weights")
    )
    return {
        "service_name": service_name,
        "cache_dir": Path(cache_dir),
    }


def _cache_key(prompts, service_model):
    payload = json.dumps(
        {
            "prompts": prompts,
            "planner_model": service_model,
            "planner_version": PROMPT_WEIGHT_PLANNER_VERSION,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return sha1(payload.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"{cache_key}.json"


def _planner_prompt(prompts):
    prompt_lines = "\n".join(f"{idx + 1}. {prompt}" for idx, prompt in enumerate(prompts))
    return f"""
你是流式视频评测中的“段权重规划器”。

输入是一条视频的多段 prompt 序列，该序列通常遵循“开头铺垫—中间发展—结尾收束”的结构，。请你根据 prompt 本身，为每一段分配 1 到 100 的整数重要性分数。

请重点考虑：
1. 段落是否承载前后关系变化的关键转折，如果有，应给予更高权重
2. 段落是否包含清晰、可检验、不可替代的动作执行、人物交互、物体转移、状态变化或任务推进，如果包含，应给予更高权重
3. 段落是否更容易暴露模型是否真正理解并执行了 prompt，如果有，应给予更高权重
4. 段落是否只是对前面的重复、铺垫或收尾，如果是，应降低权重
5. 在语义条件相近的情况下，优先考虑事件发展与高潮阶段，尊重叙事性
6. 权重应具有区分度

要求：
- 输出长度必须与输入段数一致
- 每个分数必须是 1 到 100 的整数
- 不要评价视频质量，只分析 prompt 本身
- 只输出 JSON，不要输出任何解释、markdown 或额外文字


输出格式：
{{"segment_importance":[int,int,int,int,int,int]}}

输入 prompt 序列：
{prompt_lines}
"""


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


def _response_text(response) -> str:
    """Extract the assistant payload from OpenAI-compatible responses."""
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


def _normalize_importance(raw_importance, expected_len):
    if not isinstance(raw_importance, list) or len(raw_importance) != expected_len:
        return None

    values = []
    for item in raw_importance:
        try:
            value = int(item)
        except (TypeError, ValueError):
            return None
        values.append(max(1, min(100, value)))

    total = sum(values)
    if total <= 0:
        return None
    return [value / total for value in values]


def _load_cache(cache_path: Path, expected_len: int):
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    weights = payload.get("weights")
    if (
        not isinstance(weights, list)
        or len(weights) != expected_len
        or not all(isinstance(weight, (int, float)) for weight in weights)
    ):
        return None
    return [float(weight) for weight in weights]


def _save_cache(cache_path: Path, segment_importance, weights):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "planner_version": PROMPT_WEIGHT_PLANNER_VERSION,
        "segment_importance": segment_importance,
        "weights": weights,
    }
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_prompt_segment_weights(prompts, config=None):
    if not prompts:
        return []

    planner_cfg = _planner_config(config=config)
    service_config = resolve_service_config(
        config=config,
        service_name=planner_cfg["service_name"],
    )
    cache_key = _cache_key(prompts, service_config.model)
    if cache_key in _MEMORY_CACHE:
        return _MEMORY_CACHE[cache_key]

    cache_path = _cache_path(planner_cfg["cache_dir"], cache_key)
    cached_weights = _load_cache(cache_path, len(prompts))
    if cached_weights is not None:
        _MEMORY_CACHE[cache_key] = cached_weights
        return cached_weights

    client = build_openai_compatible_client(service_config)
    retry_policy = resolve_retry_policy(
        config=config,
        service_name=planner_cfg["service_name"],
        default_max_attempts=10,
        default_initial_delay=2.0,
        default_max_delay=45.0,
        default_backoff=1.8,
    )
    label = f"prompt_weight_planner segments={len(prompts)} cache={cache_key[:8]}"

    def _request_and_parse():
        response = client.chat.completions.create(
            model=service_config.model,
            messages=[{"role": "user", "content": _planner_prompt(prompts)}],
            max_tokens=service_config.max_tokens,
            temperature=service_config.temperature,
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
        )
        raw_text = _response_text(response)
        if not raw_text:
            raise ValueError("Empty planner response content while planning weights")

        parsed = _extract_json(raw_text)
        segment_importance = parsed.get("segment_importance")
        weights = _normalize_importance(segment_importance, len(prompts))
        if weights is None:
            raise ValueError(f"Invalid segment_importance payload: {raw_text}")
        return segment_importance, weights

    segment_importance, weights = run_with_retry(
        _request_and_parse,
        label=label,
        policy=retry_policy,
        retryable=lambda exc: True,
    )

    _save_cache(cache_path, segment_importance, weights)
    _MEMORY_CACHE[cache_key] = weights
    return weights
