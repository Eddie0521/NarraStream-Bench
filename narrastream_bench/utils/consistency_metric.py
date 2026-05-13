"""Shared helpers for the stricter subject/background consistency metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
from tqdm import tqdm

from narrastream_bench.utils.aggregation import get_sample_segment_weights, weighted_mean
from narrastream_bench.utils.local_cache import build_metric_result_cache
from narrastream_bench.utils.video_primitives import load_video_frames


DEFAULT_FRAME_POSITIONS = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_CACHE_VERSION = "v2_local_p25_inter_medoid"


def _metric_config(metric_name: str, config=None) -> dict[str, Any]:
    aggregation = (config or {}).get("aggregation", {})
    return dict(aggregation.get("metrics", {}).get(metric_name, {}))


def get_consistency_settings(metric_name: str, config=None) -> dict[str, Any]:
    metric_cfg = _metric_config(metric_name, config=config)
    frame_positions = metric_cfg.get("frame_positions", DEFAULT_FRAME_POSITIONS)
    if not isinstance(frame_positions, list) or not frame_positions:
        frame_positions = list(DEFAULT_FRAME_POSITIONS)
    normalized_positions = []
    for raw_position in frame_positions:
        try:
            normalized_positions.append(float(raw_position))
        except (TypeError, ValueError):
            continue
    if not normalized_positions:
        normalized_positions = list(DEFAULT_FRAME_POSITIONS)

    return {
        "cache_version": str(metric_cfg.get("cache_version", DEFAULT_CACHE_VERSION)),
        "frame_positions": [min(1.0, max(0.0, pos)) for pos in normalized_positions],
        "intra_reduction": str(metric_cfg.get("intra_reduction", "p25")),
        "inter_reduction": str(metric_cfg.get("inter_reduction", "p25")),
        "local_weight": float(metric_cfg.get("local_weight", 0.4)),
        "inter_weight": float(metric_cfg.get("inter_weight", 0.6)),
        "inter_adjacent_weight": float(metric_cfg.get("inter_adjacent_weight", 0.4)),
        "inter_anchor_weight": float(metric_cfg.get("inter_anchor_weight", 0.6)),
    }


def _resolve_sampled_frame_indices(total_frames: int, frame_positions: list[float]) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames == 1:
        return [0]
    return sorted(
        {
            min(total_frames - 1, max(0, int(round(position * (total_frames - 1)))))
            for position in frame_positions
        }
    )


def _reduce_similarity_scores(values: list[float], reduction: str) -> float | None:
    if not values:
        return None

    normalized = [float(value) for value in values]
    mode = str(reduction).strip().lower()
    if mode == "mean":
        return sum(normalized) / len(normalized)
    if mode == "min":
        return min(normalized)
    if mode.startswith("p") and mode[1:].isdigit():
        quantile = int(mode[1:]) / 100.0
        ordered = sorted(normalized)
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * quantile
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    raise ValueError(f"Unsupported reduction: {reduction}")


def _combine_weighted_components(components: list[tuple[str, float | None, float]]) -> float | None:
    valid_scores = []
    valid_weights = []
    for _name, score, weight in components:
        if score is None or weight <= 0:
            continue
        valid_scores.append(float(score))
        valid_weights.append(float(weight))
    if not valid_scores:
        return None
    if len(valid_scores) == 1:
        return valid_scores[0]
    return weighted_mean(valid_scores, valid_weights)


def _normalize_feature_tensor(feature: torch.Tensor) -> torch.Tensor:
    # Keep representative features on CPU so cached and freshly computed
    # segments can be mixed safely during resume / partial reruns.
    tensor = feature.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    norm = torch.linalg.norm(tensor)
    if norm <= 0:
        return tensor
    return tensor / norm


def _cosine_similarity(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return max(0.0, float(torch.dot(lhs, rhs).item()))


def _build_similarity_matrix(features: list[torch.Tensor]) -> list[list[float]]:
    matrix = []
    for lhs in features:
        row = []
        for rhs in features:
            row.append(_cosine_similarity(lhs, rhs))
        matrix.append(row)
    return matrix


def _compute_segment_details(
    *,
    sampled_frame_indices: list[int],
    features: list[torch.Tensor],
    reduction: str,
) -> dict[str, Any]:
    if not features:
        return {
            "score": 0.0,
            "sampled_frame_indices": sampled_frame_indices,
            "adjacent_scores": [],
            "center_anchor_scores": [],
            "local_pair_scores": [],
            "local_pairs": [],
            "representative_relative_index": None,
            "representative_source_index": None,
            "representative_centrality_scores": [],
            "representative_feature": None,
        }

    if len(features) == 1:
        representative_feature = features[0]
        return {
            "score": 1.0,
            "sampled_frame_indices": sampled_frame_indices,
            "adjacent_scores": [],
            "center_anchor_scores": [],
            "local_pair_scores": [1.0],
            "local_pairs": [],
            "representative_relative_index": 0,
            "representative_source_index": sampled_frame_indices[0],
            "representative_centrality_scores": [1.0],
            "representative_feature": representative_feature,
        }

    similarity_matrix = _build_similarity_matrix(features)
    center_index = len(features) // 2
    seen_pairs: set[tuple[int, int]] = set()
    local_pairs = []
    adjacent_scores = []
    center_anchor_scores = []

    def _append_pair(index_a: int, index_b: int, pair_type: str) -> None:
        if index_a == index_b:
            return
        pair_key = tuple(sorted((index_a, index_b)))
        if pair_key in seen_pairs:
            return
        seen_pairs.add(pair_key)
        score = float(similarity_matrix[index_a][index_b])
        local_pairs.append(
            {
                "type": pair_type,
                "relative_indices": [index_a, index_b],
                "source_indices": [
                    sampled_frame_indices[index_a],
                    sampled_frame_indices[index_b],
                ],
                "score": score,
            }
        )
        if pair_type == "adjacent":
            adjacent_scores.append(score)
        else:
            center_anchor_scores.append(score)

    for index in range(len(features) - 1):
        _append_pair(index, index + 1, "adjacent")
    for index in range(len(features)):
        if index != center_index:
            _append_pair(center_index, index, "center_anchor")

    local_pair_scores = [entry["score"] for entry in local_pairs]
    local_score = _reduce_similarity_scores(local_pair_scores, reduction)
    if local_score is None:
        local_score = 1.0

    centrality_scores = []
    for row_index, row in enumerate(similarity_matrix):
        neighbors = [row[col_index] for col_index in range(len(row)) if col_index != row_index]
        centrality_scores.append(sum(neighbors) / len(neighbors))
    representative_relative_index = max(
        range(len(features)),
        key=lambda index: (centrality_scores[index], -abs(index - center_index)),
    )
    representative_source_index = sampled_frame_indices[representative_relative_index]

    return {
        "score": float(local_score),
        "sampled_frame_indices": sampled_frame_indices,
        "adjacent_scores": adjacent_scores,
        "center_anchor_scores": center_anchor_scores,
        "local_pair_scores": local_pair_scores,
        "local_pairs": local_pairs,
        "representative_relative_index": representative_relative_index,
        "representative_source_index": representative_source_index,
        "representative_centrality_scores": centrality_scores,
        "representative_feature": features[representative_relative_index],
    }


def _load_cached_segment_details(metric_cache, sample_id, segment_index, cache_version: str):
    payload = metric_cache.load_item_state("segments", sample_id, segment_index)
    if not isinstance(payload, dict):
        return None
    if payload.get("cache_version") != cache_version or "score" not in payload:
        return None
    feature_list = payload.get("representative_feature")
    if not isinstance(feature_list, list) or not feature_list:
        return None
    return {
        "score": float(payload["score"]),
        "sampled_frame_indices": list(payload.get("sampled_frame_indices", [])),
        "adjacent_scores": [float(value) for value in payload.get("adjacent_scores", [])],
        "center_anchor_scores": [
            float(value) for value in payload.get("center_anchor_scores", [])
        ],
        "local_pair_scores": [
            float(value) for value in payload.get("local_pair_scores", [])
        ],
        "local_pairs": list(payload.get("local_pairs", [])),
        "representative_relative_index": payload.get("representative_relative_index"),
        "representative_source_index": payload.get("representative_source_index"),
        "representative_centrality_scores": [
            float(value)
            for value in payload.get("representative_centrality_scores", [])
        ],
        "representative_feature": torch.tensor(feature_list, dtype=torch.float32),
    }


def _build_segment_cache_payload(
    *,
    cache_version: str,
    segment_details: dict[str, Any],
) -> dict[str, Any]:
    representative_feature = segment_details.get("representative_feature")
    feature_list = None
    if isinstance(representative_feature, torch.Tensor):
        feature_list = representative_feature.detach().cpu().reshape(-1).tolist()

    return {
        "cache_version": cache_version,
        "score": float(segment_details["score"]),
        "sampled_frame_indices": list(segment_details.get("sampled_frame_indices", [])),
        "adjacent_scores": list(segment_details.get("adjacent_scores", [])),
        "center_anchor_scores": list(segment_details.get("center_anchor_scores", [])),
        "local_pair_scores": list(segment_details.get("local_pair_scores", [])),
        "local_pairs": list(segment_details.get("local_pairs", [])),
        "representative_relative_index": segment_details.get("representative_relative_index"),
        "representative_source_index": segment_details.get("representative_source_index"),
        "representative_centrality_scores": list(
            segment_details.get("representative_centrality_scores", [])
        ),
        "representative_feature": feature_list,
    }


def _segment_weight_details(sample, config, segment_count: int) -> tuple[list[float], str]:
    weights = get_sample_segment_weights(sample=sample, config=config)
    if weights and len(weights) == segment_count:
        return [float(weight) for weight in weights], "planner"
    if segment_count <= 0:
        return [], "empty"
    return [1.0] * segment_count, "uniform"


def _build_sample_details(
    *,
    metric_name: str,
    sample,
    segment_payloads: list[dict[str, Any]],
    settings: dict[str, Any],
    config=None,
) -> dict[str, Any]:
    segment_scores = [float(payload["score"]) for payload in segment_payloads]
    segment_weights, segment_weight_source = _segment_weight_details(
        sample=sample,
        config=config,
        segment_count=len(segment_payloads),
    )
    if segment_scores:
        local_score = weighted_mean(segment_scores, segment_weights)
    else:
        local_score = 0.0

    adjacent_scores = []
    adjacent_pairs = []
    for segment_index in range(len(segment_payloads) - 1):
        lhs_feature = segment_payloads[segment_index]["representative_feature"]
        rhs_feature = segment_payloads[segment_index + 1]["representative_feature"]
        if not isinstance(lhs_feature, torch.Tensor) or not isinstance(rhs_feature, torch.Tensor):
            continue
        score = _cosine_similarity(lhs_feature, rhs_feature)
        adjacent_scores.append(score)
        adjacent_pairs.append(
            {
                "segment_indices": [segment_index, segment_index + 1],
                "score": score,
            }
        )

    anchor_scores = []
    anchor_pairs = []
    anchor_segment_index = None
    anchor_feature = None
    for index, payload in enumerate(segment_payloads):
        feature = payload.get("representative_feature")
        if isinstance(feature, torch.Tensor):
            anchor_feature = feature
            anchor_segment_index = index
            break
    if anchor_feature is not None and anchor_segment_index is not None:
        for index in range(anchor_segment_index + 1, len(segment_payloads)):
            feature = segment_payloads[index].get("representative_feature")
            if not isinstance(feature, torch.Tensor):
                continue
            score = _cosine_similarity(anchor_feature, feature)
            anchor_scores.append(score)
            anchor_pairs.append(
                {
                    "segment_indices": [anchor_segment_index, index],
                    "score": score,
                }
            )

    inter_adjacent_score = _reduce_similarity_scores(
        adjacent_scores,
        settings["inter_reduction"],
    )
    inter_anchor_score = _reduce_similarity_scores(
        anchor_scores,
        settings["inter_reduction"],
    )
    inter_score = _combine_weighted_components(
        [
            ("adjacent", inter_adjacent_score, settings["inter_adjacent_weight"]),
            ("anchor", inter_anchor_score, settings["inter_anchor_weight"]),
        ]
    )
    final_score = _combine_weighted_components(
        [
            ("local", local_score, settings["local_weight"]),
            ("inter", inter_score, settings["inter_weight"]),
        ]
    )
    if final_score is None:
        final_score = local_score

    return {
        "score": float(final_score),
        "cache_version": settings["cache_version"],
        "raw": {
            "metric_name": metric_name,
            "frame_positions": list(settings["frame_positions"]),
            "intra_reduction": settings["intra_reduction"],
            "inter_reduction": settings["inter_reduction"],
            "segment_weight_source": segment_weight_source,
            "segment_weights": segment_weights,
            "segment_scores": segment_scores,
            "component_weights": {
                "local_weight": settings["local_weight"],
                "inter_weight": settings["inter_weight"],
                "inter_adjacent_weight": settings["inter_adjacent_weight"],
                "inter_anchor_weight": settings["inter_anchor_weight"],
            },
            "component_scores": {
                "local": float(local_score),
                "inter_adjacent": inter_adjacent_score,
                "inter_anchor": inter_anchor_score,
                "inter": inter_score,
                "final": float(final_score),
            },
            "inter_adjacent_scores": adjacent_scores,
            "inter_anchor_scores": anchor_scores,
            "inter_adjacent_pairs": adjacent_pairs,
            "inter_anchor_pairs": anchor_pairs,
            "segments": [
                {
                    "segment_index": segment_index,
                    "segment_path": sample["segment_paths"][segment_index],
                    "local_score": float(payload["score"]),
                    "sampled_frame_indices": list(payload.get("sampled_frame_indices", [])),
                    "adjacent_scores": list(payload.get("adjacent_scores", [])),
                    "center_anchor_scores": list(payload.get("center_anchor_scores", [])),
                    "local_pair_scores": list(payload.get("local_pair_scores", [])),
                    "local_pairs": list(payload.get("local_pairs", [])),
                    "representative_relative_index": payload.get(
                        "representative_relative_index"
                    ),
                    "representative_source_index": payload.get(
                        "representative_source_index"
                    ),
                    "representative_centrality_scores": list(
                        payload.get("representative_centrality_scores", [])
                    ),
                }
                for segment_index, payload in enumerate(segment_payloads)
            ],
        },
    }


def _is_cached_sample_valid(sample_state, cache_version: str) -> bool:
    return (
        isinstance(sample_state, dict)
        and sample_state.get("cache_version") == cache_version
        and "score" in sample_state
    )


def _write_raw_metric_export(
    *,
    metric_name: str,
    metric_cache,
    eval_data,
    cache_version: str,
    run_output_path: str | None,
) -> None:
    if not run_output_path:
        return
    export_dir = Path(run_output_path) / "raw_metrics"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / f"{metric_name}.jsonl"
    with export_path.open("w", encoding="utf-8") as handle:
        for sample in eval_data:
            sample_state = metric_cache.load_sample_state(sample["sample_id"])
            if not _is_cached_sample_valid(sample_state, cache_version):
                continue
            handle.write(json.dumps(sample_state, ensure_ascii=False))
            handle.write("\n")


def compute_consistency_metric(
    *,
    metric_name: str,
    eval_data,
    device,
    config=None,
    path_config=None,
    run_output_path=None,
    build_encoder: Callable[[], Any],
    encode_selected_frames: Callable[[Any, list[Any]], list[torch.Tensor]],
    model_log_message: str,
) -> dict[Any, float]:
    settings = get_consistency_settings(metric_name, config=config)
    metric_cache = build_metric_result_cache(
        metric_name,
        run_output_path=run_output_path,
        config=config,
    )

    results = {}
    pending_samples = []
    for sample in eval_data:
        sample_state = metric_cache.load_sample_state(sample["sample_id"])
        if _is_cached_sample_valid(sample_state, settings["cache_version"]):
            results[sample["sample_id"]] = float(sample_state["score"])
        else:
            pending_samples.append(sample)

    if results:
        print(f"{metric_name} local cache hit: {len(results)}/{len(eval_data)} samples")

    if not pending_samples:
        _write_raw_metric_export(
            metric_name=metric_name,
            metric_cache=metric_cache,
            eval_data=eval_data,
            cache_version=settings["cache_version"],
            run_output_path=run_output_path,
        )
        return results

    print(model_log_message)
    encoder = build_encoder()

    sample_progress = tqdm(
        pending_samples,
        desc=f"{metric_name} (cached)",
        unit="sample",
    )
    for sample in sample_progress:
        sample_id = sample["sample_id"]
        segment_payloads = []
        total_segments = len(sample["segment_paths"])
        sample_progress.set_postfix_str(
            f"sample={sample_id}, segments={total_segments}"
        )
        for segment_index, segment_path in enumerate(sample["segment_paths"]):
            cached_segment = _load_cached_segment_details(
                metric_cache,
                sample_id=sample_id,
                segment_index=segment_index,
                cache_version=settings["cache_version"],
            )
            if cached_segment is not None:
                segment_payloads.append(cached_segment)
                continue

            frames = load_video_frames(segment_path)
            sampled_frame_indices = _resolve_sampled_frame_indices(
                len(frames),
                settings["frame_positions"],
            )
            sampled_frames = [frames[index] for index in sampled_frame_indices]
            raw_features = encode_selected_frames(encoder, sampled_frames)
            features = [_normalize_feature_tensor(feature) for feature in raw_features]
            segment_details = _compute_segment_details(
                sampled_frame_indices=sampled_frame_indices,
                features=features,
                reduction=settings["intra_reduction"],
            )
            metric_cache.save_item_state(
                "segments",
                sample_id,
                segment_index,
                _build_segment_cache_payload(
                    cache_version=settings["cache_version"],
                    segment_details=segment_details,
                ),
            )
            segment_payloads.append(segment_details)

        sample_details = _build_sample_details(
            metric_name=metric_name,
            sample=sample,
            segment_payloads=segment_payloads,
            settings=settings,
            config=config,
        )
        metric_cache.save_sample_state(sample_id, sample_details)
        results[sample_id] = float(sample_details["score"])

    _write_raw_metric_export(
        metric_name=metric_name,
        metric_cache=metric_cache,
        eval_data=eval_data,
        cache_version=settings["cache_version"],
        run_output_path=run_output_path,
    )
    return results
