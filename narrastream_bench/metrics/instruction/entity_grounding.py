"""Segment-parallel Entity Grounding with retry and frame caching."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from threading import local

from narrastream_bench.models.vlm_client import VLMClient
from tqdm import tqdm
from narrastream_bench.utils.aggregation import aggregate_metric_scores, get_metric_aggregation_config
from narrastream_bench.utils.local_cache import build_metric_result_cache, split_cached_samples
from narrastream_bench.utils.entity_extraction import extract_prompt_entity_grounding
from narrastream_bench.utils.api_retry import resolve_retry_policy, run_with_retry


_THREAD_LOCAL = local()


def _frame_cache_stats():
    cache_stats = getattr(VLMClient, "cache_stats", None)
    if callable(cache_stats):
        return cache_stats()
    return {"entries": 0, "hits": 0, "misses": 0}


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


def _normalize_score(value):
    score = float(value)
    if 0.0 <= score <= 1.0:
        return max(0.0, min(1.0, score))
    return max(0.0, min(1.0, score / 100.0))


def _build_scoring_prompt(extracted_payload):
    return f"""你是视频评分器。只看关键帧中的直接视觉证据，不要脑补剧情、常识或前后文。

任务：
对每个实体输出 `presence` 和 `attribute_match`

评分重点：
- 看身份、服装颜色、持有物、动作、状态变化、交互对象、物体归属、关系方向
- 背景相似不能加高分；多人身份不确定时不要高分
- 实体未出现则 `attribute_match=0`
- 无 attributes 的实体：出现则 `attribute_match=100`，否则为 0

分数参考：
- `presence`: 0 不存在；30 只有弱证据；50 在场但不确定；75 基本明确；100 非常明确
- `attribute_match`: 0 关键属性错；30 只对泛化属性；50 部分匹配；75 大体匹配；100 几乎完全匹配

硬约束：
- 身份不确定时，`presence` 通常不高于 60
- 动作/对象/方向/归属/状态变化错时，`attribute_match` 通常不高于 40

输出要求：
- 只输出一行紧凑 JSON
- 必须包含 `entity_results`
- `entity_results` 长度必须等于 `entities` 长度，且顺序一致
- 不要输出实体名，不要输出任何解释

输入：
{json.dumps(extracted_payload, ensure_ascii=False, separators=(",", ":"))}

示例：
{{"entity_results":[{{"presence":82,"attribute_match":74}}]}}"""


def _normalize_entity_results(entity_results, expected_entities):
    if not isinstance(entity_results, list):
        raise ValueError("entity_results must be a list")
    if len(entity_results) != len(expected_entities):
        raise ValueError(
            "entity_results length mismatch: "
            f"expected {len(expected_entities)}, got {len(entity_results)}"
        )

    normalized_results = []
    for expected_entity, raw_result in zip(expected_entities, entity_results):
        if not isinstance(raw_result, dict):
            raise ValueError("Each entity result must be an object")

        presence = _normalize_score(raw_result.get("presence"))
        if expected_entity.get("attributes"):
            attribute_match = _normalize_score(raw_result.get("attribute_match"))
        else:
            attribute_match = 1.0 if presence > 0 else 0.0

        normalized_results.append(
            {
                "name": expected_entity["name"],
                "presence": presence,
                "attribute_match": attribute_match,
            }
        )
    return normalized_results


def _parse_segment_assessment(raw_text, expected_entities):
    payload = _extract_json(raw_text)
    return _normalize_entity_results(
        payload.get("entity_results"),
        expected_entities,
    )


def _compute_entity_component(entity_results):
    if not entity_results:
        return 0.0

    entity_scores = []
    for entity_result in entity_results:
        presence = entity_result["presence"]
        attribute_match = entity_result["attribute_match"]
        entity_scores.append(presence * attribute_match)
    return sum(entity_scores) / len(entity_scores)


def _compute_segment_score(segment_entity_score):
    return segment_entity_score


def _build_segment_state(
    *,
    extracted_payload,
    raw_response,
    entity_results,
    segment_entity_score,
    segment_score,
):
    return {
        "score": segment_score,
        "raw_response": raw_response,
        "extracted_payload": extracted_payload,
        "entity_results": entity_results,
        "segment_entity_score": segment_entity_score,
        "scoring_variant": "EG-Entity",
        "entity_formula": "entity_score = presence * attribute_match",
        "score_formula": "segment_score = mean_i(presence_i * attribute_match_i)",
    }


def _score_segment_individually(
    *,
    vlm,
    segment_path,
    extracted_payload,
    expected_entities,
    frames_per_segment,
    max_tokens,
):
    raw_text = vlm.evaluate_segments(
        [segment_path],
        _build_scoring_prompt(extracted_payload),
        frames_per_segment=frames_per_segment,
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
    )
    entity_results = _parse_segment_assessment(raw_text, expected_entities)
    return raw_text, entity_results


def _get_thread_local_vlm(model, config, path_config):
    key = (model, id(config), id(path_config))
    existing_key = getattr(_THREAD_LOCAL, "vlm_key", None)
    if existing_key != key:
        _THREAD_LOCAL.vlm_client = VLMClient(
            model=model,
            config=config,
            path_config=path_config,
        )
        _THREAD_LOCAL.vlm_key = key
    return _THREAD_LOCAL.vlm_client


def _score_segment_task(
    task,
    *,
    config,
    path_config,
    vlm_model,
    frames_per_segment,
    max_tokens,
):
    vlm = _get_thread_local_vlm(vlm_model, config, path_config)
    retry_policy = resolve_retry_policy(
        config=config,
        service_name="vlm",
        default_max_attempts=12,
        default_initial_delay=2.0,
        default_max_delay=60.0,
        default_backoff=1.8,
    )
    label = (
        f"entity_grounding sample={task['sample_id']} "
        f"segment={task['index'] + 1}"
    )

    def _request_and_parse():
        raw_text, entity_results = _score_segment_individually(
            vlm=vlm,
            segment_path=task["segment_path"],
            extracted_payload=task["extracted_payload"],
            expected_entities=task["expected_entities"],
            frames_per_segment=frames_per_segment,
            max_tokens=max_tokens,
        )
        segment_entity_score = _compute_entity_component(entity_results)
        score = _compute_segment_score(segment_entity_score)
        segment_state = _build_segment_state(
            extracted_payload=task["extracted_payload"],
            raw_response=raw_text,
            entity_results=entity_results,
            segment_entity_score=segment_entity_score,
            segment_score=score,
        )
        return score, segment_state

    score, segment_state = run_with_retry(
        _request_and_parse,
        label=label,
        policy=retry_policy,
        retryable=lambda exc: not isinstance(exc, ValueError),
    )
    return task["sample_id"], task["index"], score, segment_state


def _extract_entity_task(
    task,
    *,
    config,
):
    extracted_payload = extract_prompt_entity_grounding(
        task["prompt"],
        config=config,
        request_label=(
            f"entity_extract sample={task['sample_id']} "
            f"segment={task['index'] + 1}"
        ),
    )
    return task, extracted_payload


def compute_entity_grounding(eval_data, device, config=None, path_config=None, **kwargs):
    del device
    api_workers = max(1, int(kwargs.get("api_workers", 4)))
    print(f"Initializing Entity Grounding path with api_workers={api_workers}...")
    metric_cache = build_metric_result_cache(
        "entity_grounding",
        run_output_path=kwargs.get("run_output_path"),
        config=config,
    )
    metric_cfg = get_metric_aggregation_config("entity_grounding", config)
    fallback_score = float(metric_cfg.get("fallback_score", 0.5))
    empty_entity_score = float(metric_cfg.get("empty_entity_score", fallback_score))
    frames_per_segment = (
        (config or {}).get("evaluation", {}).get("vlm_frames_per_segment", 5)
    )
    request_max_tokens = max(
        256,
        int((config or {}).get("services", {}).get("vlm", {}).get("max_tokens", 256)),
    )

    cached_results, pending_samples = split_cached_samples(eval_data, metric_cache)
    if cached_results:
        print(f"Entity Grounding local cache hit: {len(cached_results)}/{len(eval_data)} samples")

    sample_payloads = {}
    segment_tasks = []
    extraction_tasks = []
    total_pairs = sum(
        min(len(sample["prompts"]), len(sample["segment_paths"]))
        for sample in pending_samples
    )
    extraction_completed = 0
    extraction_requests = 0
    extraction_empty_entities = 0
    with tqdm(total=total_pairs, desc="Entity Extraction") as extraction_pbar:
        for sample in pending_samples:
            sample_id = sample["sample_id"]
            prompts = sample["prompts"]
            segment_paths = sample["segment_paths"]
            pair_count = min(len(prompts), len(segment_paths))
            scores = [fallback_score] * pair_count
            sample_payloads[sample_id] = {
                "sample": sample,
                "scores": scores,
            }

            for idx, (prompt, segment_path) in enumerate(
                zip(prompts[:pair_count], segment_paths[:pair_count])
            ):
                cached_segment_state = metric_cache.load_item_state(
                    "segments",
                    sample_id,
                    idx,
                )
                if isinstance(cached_segment_state, dict) and "score" in cached_segment_state:
                    sample_payloads[sample_id]["scores"][idx] = float(cached_segment_state["score"])
                    extraction_pbar.update(1)
                    extraction_completed += 1
                    extraction_requests += 1
                    continue

                extraction_tasks.append(
                    {
                        "sample_id": sample_id,
                        "index": idx,
                        "prompt": prompt,
                        "segment_path": segment_path,
                    }
                )

        with ThreadPoolExecutor(max_workers=api_workers) as executor:
            future_to_task = {
                executor.submit(
                    _extract_entity_task,
                    task,
                    config=config,
                ): task
                for task in extraction_tasks
            }
            for future in as_completed(future_to_task):
                extraction_completed += 1
                extraction_requests += 1
                extraction_pbar.update(1)
                task = future_to_task[future]
                try:
                    _, extracted_payload = future.result()
                except Exception as exc:
                    print(
                        "Entity extraction failed for sample "
                        f"{task['sample_id']} segment {task['index'] + 1}: {exc}"
                    )
                    sample_payloads[task["sample_id"]]["scores"][task["index"]] = (
                        fallback_score
                    )
                    metric_cache.save_item_state(
                        "segments",
                        task["sample_id"],
                        task["index"],
                        {
                            "score": fallback_score,
                            "error": str(exc),
                        },
                    )
                    continue

                expected_entities = extracted_payload.get("entities", [])
                if not expected_entities:
                    extraction_empty_entities += 1
                    sample_payloads[task["sample_id"]]["scores"][task["index"]] = (
                        empty_entity_score
                    )
                    metric_cache.save_item_state(
                        "segments",
                        task["sample_id"],
                        task["index"],
                        {
                            "score": empty_entity_score,
                            "reason": "empty_entities",
                            "extracted_payload": extracted_payload,
                        },
                    )
                    continue

                segment_tasks.append(
                    {
                        "sample_id": task["sample_id"],
                        "index": task["index"],
                        "segment_path": task["segment_path"],
                        "extracted_payload": extracted_payload,
                        "expected_entities": expected_entities,
                    }
                )

    results = dict(cached_results)
    segment_requests = 0
    with ThreadPoolExecutor(max_workers=api_workers) as executor:
        future_to_task = {
            executor.submit(
                _score_segment_task,
                task,
                config=config,
                path_config=path_config,
                vlm_model=kwargs.get("vlm_model"),
                frames_per_segment=frames_per_segment,
                max_tokens=request_max_tokens,
            ): task
            for task in segment_tasks
        }
        for future in tqdm(
            as_completed(future_to_task),
            total=len(future_to_task),
            desc="Entity Grounding",
        ):
            task = future_to_task[future]
            segment_requests += 1
            try:
                sample_id, segment_index, score, segment_state = future.result()
                sample_payloads[sample_id]["scores"][segment_index] = score
                metric_cache.save_item_state(
                    "segments",
                    sample_id,
                    segment_index,
                    segment_state,
                )
            except Exception as exc:
                print(
                    "Entity Grounding failed for sample "
                    f"{task['sample_id']} segment {task['index'] + 1}: {exc}"
                )
                sample_payloads[task["sample_id"]]["scores"][task["index"]] = fallback_score
                metric_cache.save_item_state(
                    "segments",
                    task["sample_id"],
                    task["index"],
                    {
                        "score": fallback_score,
                        "error": str(exc),
                        "extracted_payload": task["extracted_payload"],
                    },
                )

    for sample_id, payload in sample_payloads.items():
        sample_score = aggregate_metric_scores(
            "entity_grounding",
            payload["scores"],
            config=config,
            default_strategy="mean",
            sample=payload["sample"],
        )
        results[sample_id] = sample_score
        metric_cache.save_sample_state(
            sample_id,
            {
                "score": sample_score,
                "segment_scores": payload["scores"],
                "scoring_variant": "EG-Entity",
                "entity_formula": "presence * attribute_match",
            },
        )

    print(
        "Entity Extraction summary: "
        f"pairs={total_pairs}, "
        f"completed={extraction_completed}, "
        f"requests={extraction_requests}, "
        f"empty_entities={extraction_empty_entities}"
    )
    print(
        "Entity Grounding request summary: "
        f"api_workers={api_workers}, "
        f"segment_requests={segment_requests}, "
        f"frame_cache={_frame_cache_stats()}"
    )
    return results
