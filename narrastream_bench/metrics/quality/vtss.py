"""VTSS-based quality score with NarraStream-Bench normalization and aggregation."""
from __future__ import annotations

import json
import numbers
import os
from pathlib import Path

from tqdm import tqdm

from narrastream_bench.models.vtss_evaluator import VTSSEvaluator
from narrastream_bench.utils.aggregation import aggregate_metric_scores
from narrastream_bench.utils.local_cache import build_metric_result_cache


RAW_SCORE_VERSION = "vtss_raw_v2"


def _is_valid_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _metric_runtime_config(config=None) -> tuple[str, float, float, str]:
    metric_cfg = ((config or {}).get("aggregation") or {}).get("metrics", {}).get(
        "vtss",
        {},
    )
    strategy = str(metric_cfg.get("strategy", "vde_decay"))
    weight_type = str(metric_cfg.get("weight_type", "linear"))
    score_mapping = str(metric_cfg.get("score_mapping", "anchored_linear"))
    raw_low = float(metric_cfg.get("raw_low", 0.02))
    raw_high = float(metric_cfg.get("raw_high", 0.075))

    if raw_high <= raw_low:
        raise ValueError(
            f"vtss raw_high must be greater than raw_low, got {raw_high} <= {raw_low}"
        )

    score_version = (
        "v3_"
        f"{strategy}_"
        f"{weight_type}_"
        f"{score_mapping}_"
        f"low_{raw_low:.6f}_"
        f"high_{raw_high:.6f}"
    )
    return score_mapping, raw_low, raw_high, score_version


def _score_from_raw_vtss(
    raw_score: float,
    *,
    score_mapping: str,
    raw_low: float,
    raw_high: float,
) -> float:
    if score_mapping != "anchored_linear":
        raise ValueError(f"Unsupported vtss score_mapping: {score_mapping}")

    normalized = (float(raw_score) - raw_low) / (raw_high - raw_low)
    return max(0.0, min(1.0, normalized))


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _artifact_path(run_output_path) -> Path | None:
    if not run_output_path:
        return None
    return Path(run_output_path) / "artifacts" / "vtss_details.json"


def _save_vtss_artifact(
    eval_data,
    metric_cache,
    *,
    run_output_path,
    score_mapping: str,
    raw_low: float,
    raw_high: float,
    score_version: str,
) -> None:
    artifact_path = _artifact_path(run_output_path)
    if artifact_path is None:
        return

    samples = []
    for sample in eval_data:
        sample_state = metric_cache.load_sample_state(sample["sample_id"])
        if not isinstance(sample_state, dict):
            continue
        samples.append(
            {
                "sample_id": sample["sample_id"],
                "vtss": sample_state.get("score"),
                "vtss_raw": sample_state.get("vtss_raw"),
                "segment_raw_scores": list(sample_state.get("segment_raw_scores", [])),
            }
        )

    payload = {
        "metric": "vtss",
        "score_mapping": score_mapping,
        "raw_low": raw_low,
        "raw_high": raw_high,
        "score_version": score_version,
        "raw_score_version": RAW_SCORE_VERSION,
        "samples": samples,
    }
    _write_json_atomic(artifact_path, payload)


def compute_vtss(eval_data, device, config=None, path_config=None, **kwargs):
    run_output_path = kwargs.get("run_output_path")
    metric_cache = build_metric_result_cache(
        "vtss",
        run_output_path=run_output_path,
        config=config,
    )
    score_mapping, raw_low, raw_high, score_version = _metric_runtime_config(
        config=config
    )

    results = {}
    pending_samples = []
    for sample in eval_data:
        sample_state = metric_cache.load_sample_state(sample["sample_id"])
        if (
            isinstance(sample_state, dict)
            and _is_valid_number(sample_state.get("score"))
            and sample_state.get("score_version") == score_version
        ):
            results[sample["sample_id"]] = sample_state["score"]
            continue
        pending_samples.append(sample)

    if results:
        print(f"VTSS local cache hit: {len(results)}/{len(eval_data)} samples")
    if not pending_samples:
        _save_vtss_artifact(
            eval_data,
            metric_cache,
            run_output_path=run_output_path,
            score_mapping=score_mapping,
            raw_low=raw_low,
            raw_high=raw_high,
            score_version=score_version,
        )
        return results

    print("Loading VTSS evaluator...")
    evaluator = VTSSEvaluator(device=device, config=config, path_config=path_config)

    for sample in tqdm(pending_samples, desc="VTSS"):
        sample_id = sample["sample_id"]
        segment_raw_scores = []
        failed_segments = []
        for idx, segment_path in enumerate(sample["segment_paths"]):
            cached_item_state = metric_cache.load_item_state("segments", sample_id, idx)
            cached_raw_score = None
            if isinstance(cached_item_state, dict):
                if (
                    cached_item_state.get("score_version") == RAW_SCORE_VERSION
                    and _is_valid_number(cached_item_state.get("raw_score"))
                ):
                    cached_raw_score = float(cached_item_state["raw_score"])
                elif (
                    cached_item_state.get("score_version") == RAW_SCORE_VERSION
                    and _is_valid_number(cached_item_state.get("score"))
                ):
                    # Backward compatible with old caches where score already meant raw VTSS.
                    cached_raw_score = float(cached_item_state["score"])

            if cached_raw_score is None:
                segment_score = evaluator.score_video(segment_path)
                if segment_score is None:
                    failed_segments.append(
                        {
                            "item_id": idx,
                            "segment_path": segment_path,
                            "reason": "segment_inference_failed",
                        }
                    )
                    break
                cached_raw_score = float(segment_score)

            metric_cache.save_item_score(
                "segments",
                sample_id,
                idx,
                cached_raw_score,
                raw_score=cached_raw_score,
                score_version=RAW_SCORE_VERSION,
            )
            segment_raw_scores.append(cached_raw_score)

        if failed_segments:
            results[sample_id] = None
            metric_cache.save_sample_state(
                sample_id,
                {
                    "score": None,
                    "vtss_raw": None,
                    "segment_raw_scores": segment_raw_scores,
                    "score_mapping": score_mapping,
                    "raw_low": raw_low,
                    "raw_high": raw_high,
                    "score_version": score_version,
                    "raw_score_version": RAW_SCORE_VERSION,
                    "failure_reason": "segment_inference_failed",
                    "failed_segments": failed_segments,
                },
            )
            continue

        sample_raw_score = aggregate_metric_scores(
            "vtss",
            segment_raw_scores,
            config=config,
            default_strategy="vde_decay",
            sample=sample,
        )
        sample_score = _score_from_raw_vtss(
            sample_raw_score,
            score_mapping=score_mapping,
            raw_low=raw_low,
            raw_high=raw_high,
        )
        results[sample_id] = sample_score
        metric_cache.save_sample_score(
            sample_id,
            sample_score,
            vtss_raw=sample_raw_score,
            segment_raw_scores=segment_raw_scores,
            score_mapping=score_mapping,
            raw_low=raw_low,
            raw_high=raw_high,
            score_version=score_version,
            raw_score_version=RAW_SCORE_VERSION,
        )

    _save_vtss_artifact(
        eval_data,
        metric_cache,
        run_output_path=run_output_path,
        score_mapping=score_mapping,
        raw_low=raw_low,
        raw_high=raw_high,
        score_version=score_version,
    )
    return results
