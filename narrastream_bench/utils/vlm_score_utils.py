"""Shared helpers for VLM score parsing, weighting, and sample persistence."""

from __future__ import annotations

import json

from narrastream_bench.utils.aggregation import get_sample_segment_weights, weighted_mean


VLM_SCORE_VERSION = "v7_prompt_state_continuity_identity_strict_1_100"

PROMPT_TEMPLATE = """你是一个视频评估专家。下面是一个流式生成的视频和对应的提示词序列。
系统会按时间顺序提供每一段视频的关键帧。请判断每一段视频是否执行了对应的 prompt，并给出整条序列的整体评分。

提示词序列：
{prompts}

要求：
1. `segment_scores` 必须是长度为 {num_segments} 的数组，第 i 个元素对应第 i 段 prompt 的执行分。
2. 所有分数都使用 1-100 的整数。1 表示完全不符合，100 表示完全符合。
3. 默认根据“该段关键帧”和“该段 prompt”评分，不要根据开头/中间/结尾位置、叙事结构或段落重要性加分。
4. 如果该段 prompt 主要描述的是延续、保持、确认、观察、仍然处于某状态，可以结合前一段已经建立的状态判断该状态是否持续成立。不要因为该段动作幅度较小，就机械给低分。
5. 必须依据可直接观察到的证据评分：
   - 明确动作是否发生
   - 人物交互是否发生
   - 物体变化、物体 handoff 或状态转移是否发生
   - 是否在正确时间点执行了该段 prompt
   - 如果该段是延续/保持类 prompt，是否清楚地维持了前序已建立的人物关系、物体状态和空间布局，且没有明显冲突
6. 如果只匹配了人物、场景或大致氛围，但关键动作、交互、物体归属、handoff 接收关系或状态变化并不明确，`segment_score` 不应高于 50。
   对于多人场景，只要人物身份、交互对象或物体归属存在明显不确定性，即使画面整体相似，也不应给高分。
7. 如果几乎没有直接证据支持该段 prompt，`segment_score` 应在 1-20。
   如果只存在弱相关、泛化、可替代的证据，`segment_score` 通常应在 20-50，而不是高分。
8. 允许相邻段同分，也允许大幅跳变；不要为了看起来平滑而输出递增或递减的分数序列。
   但也不要因为后续段动作更弱，就自动把“延续成立”的段落打成极低分。
9. `overall_score` 表示整条视频是否按正确时间顺序完成了关键事件并维持了后续状态一致性。
   如果关键动作、主要交互、物体 handoff 或状态转折已经清楚发生，且后续没有明显违背 prompt，即使后续段落动作较弱，也可以给中高分。
   反过来，如果只是整体流畅、场景一致、人物大致相似，但关键事件、人物关系或物体归属无法确认，`overall_score` 应明显降低。
10. 只有当人物身份、关键动作、物体关系和时间点都非常明确且几乎无歧义时，才给 100 分。

输出格式：
{{"segment_scores":[int,int,...],"overall_score":int}}"""


def build_vlm_prompt_text(prompts):
    return PROMPT_TEMPLATE.format(
        prompts="\n".join([f"{i + 1}. {prompt}" for i, prompt in enumerate(prompts)]),
        num_segments=len(prompts),
    )


def _extract_json(text):
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


def _normalize_percentage_score(value):
    score = float(value)
    return max(0.0, min(1.0, score / 100.0))


def parse_legacy_score(raw_text):
    score = float(raw_text.strip())
    if 0.0 <= score <= 1.0:
        return score
    score = (score - 1.0) / 4.0
    return max(0.0, min(1.0, score))


def parse_structured_scores(raw_text, expected_segments):
    payload = _extract_json(raw_text)

    segment_scores = payload.get("segment_scores")
    overall_score = payload.get("overall_score")

    if not isinstance(segment_scores, list):
        raise ValueError("segment_scores must be a list")
    if len(segment_scores) != expected_segments:
        raise ValueError(
            f"segment_scores length mismatch: expected {expected_segments}, got {len(segment_scores)}"
        )
    if overall_score is None:
        raise ValueError("overall_score is required")

    normalized_segments = [_normalize_percentage_score(value) for value in segment_scores]
    normalized_overall = _normalize_percentage_score(overall_score)
    return normalized_segments, normalized_overall


def compute_segment_component(
    segment_scores,
    *,
    sample=None,
    config=None,
    fallback_score=0.5,
):
    if not segment_scores:
        return fallback_score, None, "empty_fallback"

    prompt_weights = get_sample_segment_weights(sample=sample, config=config)
    if prompt_weights and len(prompt_weights) == len(segment_scores):
        try:
            return (
                weighted_mean(segment_scores, prompt_weights),
                [float(weight) for weight in prompt_weights],
                "llm_prompt_weighted_mean",
            )
        except (TypeError, ValueError):
            pass

    return sum(segment_scores) / len(segment_scores), None, "mean_fallback"


def compute_weighted_vlm_score(
    segment_scores,
    overall_score,
    *,
    sample=None,
    config=None,
    segment_weight,
    overall_weight,
    fallback_score,
):
    segment_component, prompt_weights, segment_weighting_mode = compute_segment_component(
        segment_scores,
        sample=sample,
        config=config,
        fallback_score=fallback_score,
    )
    weighted_components = []
    if segment_weight > 0:
        weighted_components.append((segment_component, segment_weight))
    if overall_weight > 0:
        weighted_components.append((overall_score, overall_weight))

    if not weighted_components:
        score = fallback_score
    else:
        total_weight = sum(weight for _, weight in weighted_components)
        score = sum(value * weight for value, weight in weighted_components) / total_weight

    return score, segment_component, prompt_weights, segment_weighting_mode


def build_vlm_sample_state(
    *,
    score,
    raw_response=None,
    parse_mode=None,
    segment_scores=None,
    overall_score=None,
    segment_component=None,
    segment_prompt_weights=None,
    segment_weighting_mode=None,
    segment_weight=None,
    overall_weight=None,
    frames_per_segment=None,
    num_segments=None,
    error=None,
    score_version=VLM_SCORE_VERSION,
):
    payload = {
        "score": score,
        "score_version": score_version,
    }
    if raw_response is not None:
        payload["raw_response"] = raw_response
    if parse_mode is not None:
        payload["parse_mode"] = parse_mode
    if segment_scores is not None:
        payload["segment_scores"] = segment_scores
    if overall_score is not None:
        payload["overall_score"] = overall_score
    if segment_component is not None:
        payload["segment_component"] = segment_component
    if segment_prompt_weights is not None:
        payload["segment_prompt_weights"] = segment_prompt_weights
    if segment_weighting_mode is not None:
        payload["segment_weighting_mode"] = segment_weighting_mode
    if segment_weight is not None:
        payload["segment_weight"] = segment_weight
    if overall_weight is not None:
        payload["overall_weight"] = overall_weight
    if frames_per_segment is not None:
        payload["frames_per_segment"] = frames_per_segment
    if num_segments is not None:
        payload["num_segments"] = num_segments
    if error is not None:
        payload["error"] = error
    return payload


def parse_vlm_response_with_details(
    raw_response,
    *,
    expected_segments,
    sample=None,
    config=None,
    segment_weight,
    overall_weight,
    fallback_score,
):
    try:
        segment_scores, overall_score = parse_structured_scores(
            raw_response,
            expected_segments=expected_segments,
        )
        score, segment_component, prompt_weights, segment_weighting_mode = (
            compute_weighted_vlm_score(
                segment_scores,
                overall_score,
                sample=sample,
                config=config,
                segment_weight=segment_weight,
                overall_weight=overall_weight,
                fallback_score=fallback_score,
            )
        )
        return build_vlm_sample_state(
            score=score,
            raw_response=raw_response,
            parse_mode="structured_json",
            segment_scores=segment_scores,
            overall_score=overall_score,
            segment_component=segment_component,
            segment_prompt_weights=prompt_weights,
            segment_weighting_mode=segment_weighting_mode,
            segment_weight=segment_weight,
            overall_weight=overall_weight,
            num_segments=expected_segments,
        )
    except Exception:
        score = parse_legacy_score(raw_response)
        return build_vlm_sample_state(
            score=score,
            raw_response=raw_response,
            parse_mode="legacy_score",
            segment_weighting_mode="legacy_direct",
            segment_weight=segment_weight,
            overall_weight=overall_weight,
            num_segments=expected_segments,
        )


def upgrade_vlm_cache_state(
    sample,
    sample_state,
    *,
    config=None,
    segment_weight,
    overall_weight,
    fallback_score,
):
    raw_response = sample_state.get("raw_response")
    if not raw_response:
        return None

    refreshed_state = parse_vlm_response_with_details(
        raw_response,
        expected_segments=len(sample.get("prompts") or []),
        sample=sample,
        config=config,
        segment_weight=segment_weight,
        overall_weight=overall_weight,
        fallback_score=fallback_score,
    )
    if "frames_per_segment" in sample_state:
        refreshed_state.setdefault(
            "frames_per_segment",
            sample_state.get("frames_per_segment"),
        )
    if "num_segments" in sample_state:
        refreshed_state.setdefault("num_segments", sample_state.get("num_segments"))
    return refreshed_state


def split_vlm_cached_samples(
    eval_data,
    metric_cache,
    *,
    config=None,
    segment_weight,
    overall_weight,
    fallback_score,
):
    cached_results = {}
    pending_samples = []
    upgraded_samples = 0

    for sample in eval_data:
        sample_id = sample["sample_id"]
        sample_state = metric_cache.load_sample_state(sample_id)
        if not isinstance(sample_state, dict) or "score" not in sample_state:
            pending_samples.append(sample)
            continue

        if sample_state.get("score_version") == VLM_SCORE_VERSION:
            cached_results[sample_id] = sample_state["score"]
            continue

        refreshed_state = upgrade_vlm_cache_state(
            sample,
            sample_state,
            config=config,
            segment_weight=segment_weight,
            overall_weight=overall_weight,
            fallback_score=fallback_score,
        )
        if refreshed_state is None:
            pending_samples.append(sample)
            continue

        cache_extra = {key: value for key, value in refreshed_state.items() if key != "score"}
        metric_cache.save_sample_score(sample_id, refreshed_state["score"], **cache_extra)
        cached_results[sample_id] = refreshed_state["score"]
        upgraded_samples += 1

    return cached_results, pending_samples, {"upgraded_samples": upgraded_samples}
