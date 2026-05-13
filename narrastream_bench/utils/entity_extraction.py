"""Entity grounding extraction helpers."""

from __future__ import annotations

from hashlib import sha1
import json
import os
from pathlib import Path
from threading import Lock, get_ident

from narrastream_bench.utils.api_clients import build_openai_compatible_client, resolve_service_config
from narrastream_bench.utils.api_retry import resolve_retry_policy, run_with_retry
from narrastream_bench.utils.runtime_dependencies import NARRASTREAM_BENCH_ROOT, resolve_repo_path


ENTITY_GROUNDING_EXTRACTOR_VERSION = "v4_entity_only"
_MEMORY_CACHE: dict[str, dict[str, object]] = {}
_CACHE_LOCK = Lock()


def _metric_config(config=None):
    metric_cfg = ((config or {}).get("aggregation") or {}).get("metrics", {})
    entity_cfg = metric_cfg.get("entity_grounding", {})
    cache_dir = (
        resolve_repo_path(entity_cfg.get("cache_dir"))
        or (NARRASTREAM_BENCH_ROOT / "cache" / "entity_grounding")
    )
    return {
        "service_name": entity_cfg.get("llm_service_name", "planner"),
        "cache_dir": Path(cache_dir),
        "max_entities": max(1, int(entity_cfg.get("max_entities", 4))),
        "max_attributes_per_entity": max(
            0,
            int(entity_cfg.get("max_attributes_per_entity", 5)),
        ),
    }


def _cache_key(
    prompt,
    service_model,
    max_entities,
    max_attributes_per_entity,
):
    payload = json.dumps(
        {
            "prompt": prompt,
            "service_model": service_model,
            "extractor_version": ENTITY_GROUNDING_EXTRACTOR_VERSION,
            "max_entities": max_entities,
            "max_attributes_per_entity": max_attributes_per_entity,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return sha1(payload.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"{cache_key}.json"


def _extractor_prompt(
    prompt,
    max_entities,
    max_attributes_per_entity,
):
    return f"""你是视频评测中的实体抽取器。

请从下面这条单段视频 prompt 中提取：
“可视觉验证”的主体实体，以及每个实体对应的可见属性

要求：
1. 只保留画面中可以直接验证的实体。
2. attributes 只保留绑定在该实体上的可见属性，例如：衣服、颜色、表情、动作、姿态、持有物、状态。
3. 不要输出抽象风格词、审美词、画质词，例如：beautiful、cinematic、high quality。
4. 优先保留关键人物、被操作物体、交互对象。除非环境本身是动作直接作用对象，否则不要把 room、table、lamp、window light 这类静态背景项作为实体输出。
5. 最多输出 {max_entities} 个实体；每个实体最多输出 {max_attributes_per_entity} 个 attributes。
6. 只输出 JSON，不要输出任何解释、markdown 或额外文字。

输出格式：
{{
  "entities": [
    {{
      "name": "entity name",
      "attributes": ["attr 1", "attr 2"]
    }}
  ]
}}

输入 prompt：
{prompt}
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


def _normalize_text(value):
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    return normalized or None


def _normalize_attributes(raw_attributes, max_attributes_per_entity):
    if raw_attributes is None:
        return []

    if isinstance(raw_attributes, str):
        raw_attributes = [raw_attributes]

    if not isinstance(raw_attributes, list):
        return []

    normalized = []
    seen = set()
    for attribute in raw_attributes:
        text = _normalize_text(attribute)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
        if len(normalized) >= max_attributes_per_entity:
            break
    return normalized


def _normalize_payload(
    payload,
    max_entities,
    max_attributes_per_entity,
):
    if not isinstance(payload, dict):
        return None

    raw_entities = payload.get("entities")
    if not isinstance(raw_entities, list):
        return None

    normalized_entities = []
    entity_index = {}
    for raw_entity in raw_entities:
        if isinstance(raw_entity, str):
            name = _normalize_text(raw_entity)
            attributes = []
        elif isinstance(raw_entity, dict):
            name = _normalize_text(raw_entity.get("name") or raw_entity.get("entity"))
            attributes = _normalize_attributes(
                raw_entity.get("attributes"),
                max_attributes_per_entity,
            )
        else:
            continue

        if not name:
            continue

        key = name.casefold()
        if key in entity_index:
            existing = normalized_entities[entity_index[key]]
            seen_attributes = {attribute.casefold() for attribute in existing["attributes"]}
            for attribute in attributes:
                attribute_key = attribute.casefold()
                if attribute_key in seen_attributes:
                    continue
                existing["attributes"].append(attribute)
                seen_attributes.add(attribute_key)
                if len(existing["attributes"]) >= max_attributes_per_entity:
                    break
            continue

        if len(normalized_entities) >= max_entities:
            break

        entity_index[key] = len(normalized_entities)
        normalized_entities.append(
            {
                "name": name,
                "attributes": attributes[:max_attributes_per_entity],
            }
        )

    return {"entities": normalized_entities}


def _load_cache(cache_path: Path, max_entities, max_attributes_per_entity):
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    return _normalize_payload(
        payload,
        max_entities=max_entities,
        max_attributes_per_entity=max_attributes_per_entity,
    )


def _save_cache(cache_path: Path, payload):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(
        f"{cache_path.suffix}.tmp.{os.getpid()}.{get_ident()}"
    )
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, cache_path)


def extract_prompt_entity_grounding(prompt, config=None, request_label=None):
    prompt = (prompt or "").strip()
    if not prompt:
        return {"entities": []}

    metric_cfg = _metric_config(config=config)
    service_config = resolve_service_config(
        config=config,
        service_name=metric_cfg["service_name"],
    )

    cache_key = _cache_key(
        prompt,
        service_config.model,
        metric_cfg["max_entities"],
        metric_cfg["max_attributes_per_entity"],
    )
    with _CACHE_LOCK:
        if cache_key in _MEMORY_CACHE:
            return _MEMORY_CACHE[cache_key]

    cache_path = _cache_path(metric_cfg["cache_dir"], cache_key)
    cached_payload = _load_cache(
        cache_path,
        max_entities=metric_cfg["max_entities"],
        max_attributes_per_entity=metric_cfg["max_attributes_per_entity"],
    )
    if cached_payload is not None:
        with _CACHE_LOCK:
            _MEMORY_CACHE[cache_key] = cached_payload
        return cached_payload

    client = build_openai_compatible_client(service_config)
    retry_policy = resolve_retry_policy(
        config=config,
        service_name=metric_cfg["service_name"],
        default_max_attempts=10,
        default_initial_delay=2.0,
        default_max_delay=45.0,
        default_backoff=1.8,
    )

    label = request_label or f"entity_extract cache={cache_key[:8]}"

    def _request_and_parse():
        response = client.chat.completions.create(
            model=service_config.model,
            messages=[
                {
                    "role": "user",
                    "content": _extractor_prompt(
                        prompt,
                        max_entities=metric_cfg["max_entities"],
                        max_attributes_per_entity=metric_cfg["max_attributes_per_entity"],
                    ),
                }
            ],
            max_tokens=service_config.max_tokens,
            temperature=service_config.temperature,
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
        )
        response_text = _response_text(response)
        if not response_text:
            raise ValueError(
                "Empty planner response content while extracting entities"
            )
        normalized_payload = _normalize_payload(
            _extract_json(response_text),
            max_entities=metric_cfg["max_entities"],
            max_attributes_per_entity=metric_cfg["max_attributes_per_entity"],
        )
        if normalized_payload is None:
            raise ValueError("Invalid entity grounding payload")
        return normalized_payload

    normalized_payload = run_with_retry(
        _request_and_parse,
        label=label,
        policy=retry_policy,
        retryable=lambda exc: True,
    )
    _save_cache(cache_path, normalized_payload)
    with _CACHE_LOCK:
        _MEMORY_CACHE[cache_key] = normalized_payload
    return normalized_payload
