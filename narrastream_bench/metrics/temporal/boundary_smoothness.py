"""Boundary smoothness from single-level relative flow matching around each seam."""
from copy import deepcopy
import json
import math
from pathlib import Path

from tqdm import tqdm

from narrastream_bench.models.raft_flow import RAFTFlow
from narrastream_bench.utils.aggregation import (
    aggregate_metric_scores,
    aggregate_scores_with_explicit_weights,
    get_metric_aggregation_config,
    get_transition_subset_weights,
)
from narrastream_bench.utils.local_cache import build_metric_result_cache, split_cached_samples
from narrastream_bench.utils.video_io import load_segments


DEFAULT_CACHE_VERSION = "single_level_relative_match_v1"


def _ensure_metric_cache_version(config, metric_name, default_cache_version):
    config_copy = deepcopy(config or {})
    aggregation_cfg = config_copy.setdefault("aggregation", {})
    metrics_cfg = aggregation_cfg.setdefault("metrics", {})
    metric_cfg = metrics_cfg.setdefault(metric_name, {})
    metric_cfg.setdefault("cache_version", default_cache_version)
    return config_copy


def _window_frames(frames, window_size, take_from_end=False):
    if not frames:
        return []
    return frames[-window_size:] if take_from_end else frames[:window_size]


def _flow_magnitude(raft, frame_a, frame_b, flow_cache):
    key = (id(frame_a), id(frame_b))
    if key not in flow_cache:
        flow = raft.compute_flow(frame_a, frame_b)
        flow_cache[key] = float(flow.norm(dim=0).mean().item())
    return flow_cache[key]


def _load_cached_transition_payload(metric_cache, sample_id, transition_index):
    payload = metric_cache.load_item_state("transitions", sample_id, transition_index)
    if not isinstance(payload, dict) or "score" not in payload:
        return None

    required_keys = (
        "boundary_flow_mean",
        "left_local_flow_mean",
        "right_local_flow_mean",
        "expected_flow_mean",
        "flow_error_abs",
        "flow_tolerance",
        "normalized_flow_error",
    )
    if any(key not in payload for key in required_keys):
        return None

    return {
        "score": float(payload["score"]),
        "boundary_flow_mean": float(payload["boundary_flow_mean"]),
        "left_local_flow_mean": float(payload["left_local_flow_mean"]),
        "right_local_flow_mean": float(payload["right_local_flow_mean"]),
        "expected_flow_mean": float(payload["expected_flow_mean"]),
        "flow_error_abs": float(payload["flow_error_abs"]),
        "flow_tolerance": float(payload["flow_tolerance"]),
        "normalized_flow_error": float(payload["normalized_flow_error"]),
    }


def _compute_transition_score(
    raft,
    prev_window,
    next_window,
    *,
    flow_tolerance_abs,
    flow_tolerance_rel,
    flow_cache,
):
    if len(prev_window) < 2 or len(next_window) < 2:
        return None

    left_local_flow_mean = _flow_magnitude(
        raft,
        prev_window[-2],
        prev_window[-1],
        flow_cache,
    )
    boundary_flow_mean = _flow_magnitude(
        raft,
        prev_window[-1],
        next_window[0],
        flow_cache,
    )
    right_local_flow_mean = _flow_magnitude(
        raft,
        next_window[0],
        next_window[1],
        flow_cache,
    )

    expected_flow_mean = 0.5 * (left_local_flow_mean + right_local_flow_mean)
    flow_tolerance = float(flow_tolerance_abs) + float(flow_tolerance_rel) * float(
        expected_flow_mean
    )
    flow_tolerance = max(1e-6, flow_tolerance)
    flow_error_abs = abs(boundary_flow_mean - expected_flow_mean)
    normalized_flow_error = flow_error_abs / flow_tolerance
    transition_score = math.exp(-normalized_flow_error)

    return {
        "score": float(transition_score),
        "boundary_flow_mean": float(boundary_flow_mean),
        "left_local_flow_mean": float(left_local_flow_mean),
        "right_local_flow_mean": float(right_local_flow_mean),
        "expected_flow_mean": float(expected_flow_mean),
        "flow_error_abs": float(flow_error_abs),
        "flow_tolerance": float(flow_tolerance),
        "normalized_flow_error": float(normalized_flow_error),
    }


def _build_transition_records(
    *,
    selected_transition_indices,
    transition_payloads,
    transition_weights,
):
    records = []
    for position, transition_index in enumerate(selected_transition_indices):
        payload = transition_payloads[transition_index]
        weight = None
        if transition_weights is not None and position < len(transition_weights):
            weight = float(transition_weights[position])
        records.append(
            {
                "transition_index": int(transition_index),
                "transition_score": float(payload["score"]),
                "boundary_flow_mean": float(payload["boundary_flow_mean"]),
                "left_local_flow_mean": float(payload["left_local_flow_mean"]),
                "right_local_flow_mean": float(payload["right_local_flow_mean"]),
                "expected_flow_mean": float(payload["expected_flow_mean"]),
                "flow_error_abs": float(payload["flow_error_abs"]),
                "flow_tolerance": float(payload["flow_tolerance"]),
                "normalized_flow_error": float(payload["normalized_flow_error"]),
                "weight": weight,
            }
        )
    return records


def _build_sample_raw_metric_data(
    *,
    total_transition_count,
    selected_transition_indices,
    transition_payloads,
    transition_weights,
    strategy,
    fallback_strategy,
    window_size,
    flow_tolerance_abs,
    flow_tolerance_rel,
):
    selected_set = set(selected_transition_indices)
    return {
        "transition_records": _build_transition_records(
            selected_transition_indices=selected_transition_indices,
            transition_payloads=transition_payloads,
            transition_weights=transition_weights,
        ),
        "selected_transition_indices": [int(index) for index in selected_transition_indices],
        "skipped_transition_indices": [
            index for index in range(total_transition_count) if index not in selected_set
        ],
        "total_transition_count": int(total_transition_count),
        "valid_transition_count": int(len(selected_transition_indices)),
        "transition_weights": (
            None if transition_weights is None else [float(weight) for weight in transition_weights]
        ),
        "aggregation_strategy": strategy,
        "fallback_strategy": fallback_strategy,
        "window_size": int(window_size),
        "flow_tolerance_abs": float(flow_tolerance_abs),
        "flow_tolerance_rel": float(flow_tolerance_rel),
    }


def _write_raw_metric_export(*, metric_cache, eval_data, run_output_path, cache_version):
    if not run_output_path:
        return
    export_dir = Path(run_output_path) / "raw_metrics"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "boundary_smoothness.jsonl"
    with export_path.open("w", encoding="utf-8") as handle:
        for sample in eval_data:
            sample_state = metric_cache.load_sample_state(sample["sample_id"])
            if not isinstance(sample_state, dict) or "score" not in sample_state:
                continue
            if sample_state.get("cache_version") != cache_version:
                continue
            handle.write(json.dumps(sample_state, ensure_ascii=False))
            handle.write("\n")


def compute_boundary_smoothness(eval_data, device, config=None, path_config=None, **kwargs):
    cache_config = _ensure_metric_cache_version(
        config,
        "boundary_smoothness",
        default_cache_version=DEFAULT_CACHE_VERSION,
    )
    metric_cache = build_metric_result_cache(
        "boundary_smoothness",
        run_output_path=kwargs.get("run_output_path"),
        config=cache_config,
    )
    metric_cfg = get_metric_aggregation_config("boundary_smoothness", cache_config)
    cache_version = str(metric_cfg.get("cache_version", DEFAULT_CACHE_VERSION))
    strategy = metric_cfg.get("strategy", "mean")
    fallback_strategy = metric_cfg.get("fallback_strategy", "mean")
    window_size = max(2, int(metric_cfg.get("window_size", 2)))
    flow_tolerance_abs = float(metric_cfg.get("flow_tolerance_abs", 0.02))
    flow_tolerance_rel = float(metric_cfg.get("flow_tolerance_rel", 0.5))

    results, pending_samples = split_cached_samples(eval_data, metric_cache)
    if results:
        print(f"Boundary Smoothness local cache hit: {len(results)}/{len(eval_data)} samples")
    if not pending_samples:
        _write_raw_metric_export(
            metric_cache=metric_cache,
            eval_data=eval_data,
            run_output_path=kwargs.get("run_output_path"),
            cache_version=cache_version,
        )
        return results

    print("Loading RAFT model...")
    raft = RAFTFlow(device=device, config=config, path_config=path_config)

    for sample in tqdm(pending_samples, desc="Boundary Smoothness"):
        sample_id = sample["sample_id"]
        segment_paths = sample["segment_paths"]
        scores = []
        selected_transition_indices = []
        missing_transition_indices = []
        transition_payloads = {}

        for i in range(len(segment_paths) - 1):
            cached_transition_payload = _load_cached_transition_payload(
                metric_cache,
                sample_id,
                i,
            )
            if cached_transition_payload is not None:
                transition_payloads[i] = cached_transition_payload
            else:
                missing_transition_indices.append(i)

        segments = (
            load_segments(segment_paths, path_config=path_config)
            if missing_transition_indices
            else None
        )
        flow_cache = {}

        for i in range(len(segment_paths) - 1):
            if i in transition_payloads:
                scores.append(transition_payloads[i]["score"])
                selected_transition_indices.append(i)
                continue

            if segments is None:
                continue

            prev_window = _window_frames(segments[i], window_size, take_from_end=True)
            next_window = _window_frames(segments[i + 1], window_size, take_from_end=False)
            if not prev_window or not next_window:
                continue

            transition_payload = _compute_transition_score(
                raft,
                prev_window,
                next_window,
                flow_tolerance_abs=flow_tolerance_abs,
                flow_tolerance_rel=flow_tolerance_rel,
                flow_cache=flow_cache,
            )
            if transition_payload is None:
                continue

            metric_cache.save_item_score(
                "transitions",
                sample_id,
                i,
                transition_payload["score"],
                **{key: value for key, value in transition_payload.items() if key != "score"},
            )
            transition_payloads[i] = dict(transition_payload)
            scores.append(transition_payload["score"])
            selected_transition_indices.append(i)

        transition_weights = None
        if not scores:
            results[sample_id] = None
            metric_cache.save_sample_state(
                sample_id,
                {
                    "score": None,
                    "cache_version": cache_version,
                    "transition_indices": selected_transition_indices,
                    "raw_metric_data": _build_sample_raw_metric_data(
                        total_transition_count=max(0, len(segment_paths) - 1),
                        selected_transition_indices=selected_transition_indices,
                        transition_payloads=transition_payloads,
                        transition_weights=transition_weights,
                        strategy=strategy,
                        fallback_strategy=fallback_strategy,
                        window_size=window_size,
                        flow_tolerance_abs=flow_tolerance_abs,
                        flow_tolerance_rel=flow_tolerance_rel,
                    ),
                },
            )
            continue

        if strategy == "llm_prompt_transition_weighted_mean":
            transition_weights = get_transition_subset_weights(
                sample=sample,
                config=cache_config,
                indices=selected_transition_indices,
            )
            sample_score = aggregate_scores_with_explicit_weights(
                scores,
                transition_weights,
                fallback_strategy=fallback_strategy,
                metric_name="boundary_smoothness",
                sample=sample,
                config=cache_config,
            )
        else:
            sample_score = aggregate_metric_scores(
                "boundary_smoothness",
                scores,
                config=cache_config,
                default_strategy="mean",
                sample=sample,
            )

        results[sample_id] = sample_score
        metric_cache.save_sample_state(
            sample_id,
            {
                "score": sample_score,
                "cache_version": cache_version,
                "transition_indices": selected_transition_indices,
                "raw_metric_data": _build_sample_raw_metric_data(
                    total_transition_count=max(0, len(segment_paths) - 1),
                    selected_transition_indices=selected_transition_indices,
                    transition_payloads=transition_payloads,
                    transition_weights=transition_weights,
                    strategy=strategy,
                    fallback_strategy=fallback_strategy,
                    window_size=window_size,
                    flow_tolerance_abs=flow_tolerance_abs,
                    flow_tolerance_rel=flow_tolerance_rel,
                ),
            },
        )

    _write_raw_metric_export(
        metric_cache=metric_cache,
        eval_data=eval_data,
        run_output_path=kwargs.get("run_output_path"),
        cache_version=cache_version,
    )
    return results
