"""Motion smoothness based on AMT-S frame interpolation error."""
from __future__ import annotations

import math
import numbers

from tqdm import tqdm

from narrastream_bench.models.amt_motion import AMTMotionSmoothness
from narrastream_bench.utils.aggregation import aggregate_metric_scores
from narrastream_bench.utils.local_cache import build_metric_result_cache


def _is_valid_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _metric_runtime_config(config=None) -> tuple[str, float, str]:
    metric_cfg = ((config or {}).get("aggregation") or {}).get("metrics", {}).get(
        "motion_smoothness",
        {}
    )
    score_mapping = str(metric_cfg.get("score_mapping", "exp"))
    tau = float(metric_cfg.get("tau", 3.5))
    if tau <= 0:
        raise ValueError(f"motion_smoothness tau must be positive, got {tau}")
    score_version = f"v2_{score_mapping}_tau_{tau:.6f}"
    return score_mapping, tau, score_version


def _score_from_raw_error(raw_error: float, *, score_mapping: str, tau: float) -> float:
    if score_mapping != "exp":
        raise ValueError(
            f"Unsupported motion_smoothness score_mapping: {score_mapping}"
        )
    return float(math.exp(-float(raw_error) / tau))


def compute_motion_smoothness(eval_data, device, config=None, path_config=None, **kwargs):
    metric_cache = build_metric_result_cache(
        "motion_smoothness",
        run_output_path=kwargs.get("run_output_path"),
        config=config,
    )
    score_mapping, tau, score_version = _metric_runtime_config(config=config)

    results = {}
    pending_samples = []
    for sample in eval_data:
        sample_state = metric_cache.load_sample_state(sample["sample_id"])
        if (
            isinstance(sample_state, dict)
            and "score" in sample_state
            and sample_state.get("score_version") == score_version
        ):
            results[sample["sample_id"]] = sample_state["score"]
            continue
        pending_samples.append(sample)

    if results:
        print(
            f"Motion Smoothness local cache hit: {len(results)}/{len(eval_data)} samples"
        )
    if not pending_samples:
        return results

    print("Loading AMT model for Motion Smoothness...")
    motion = AMTMotionSmoothness(device=device, config=config, path_config=path_config)

    for sample in tqdm(pending_samples, desc="Motion Smoothness"):
        sample_id = sample["sample_id"]
        scores = []
        raw_errors = []
        for idx, segment_path in enumerate(sample["segment_paths"]):
            cached_item_state = metric_cache.load_item_state("segments", sample_id, idx)
            if isinstance(cached_item_state, dict):
                cached_raw_error = cached_item_state.get("raw_error")
                if _is_valid_number(cached_raw_error):
                    score = _score_from_raw_error(
                        float(cached_raw_error),
                        score_mapping=score_mapping,
                        tau=tau,
                    )
                    metric_cache.save_item_score(
                        "segments",
                        sample_id,
                        idx,
                        score,
                        raw_error=float(cached_raw_error),
                        score_mapping=score_mapping,
                        tau=tau,
                        score_version=score_version,
                    )
                    scores.append(score)
                    raw_errors.append(float(cached_raw_error))
                    continue

                cached_segment_score = cached_item_state.get("score")
                if (
                    cached_item_state.get("score_version") == score_version
                    and _is_valid_number(cached_segment_score)
                ):
                    scores.append(float(cached_segment_score))
                    continue

            if hasattr(motion, "evaluate_video"):
                evaluation = motion.evaluate_video(segment_path)
            else:
                evaluation = {"score": motion.score_video(segment_path)}

            score = float(evaluation["score"])
            cache_extra = {
                "score_mapping": score_mapping,
                "tau": tau,
                "score_version": score_version,
            }
            raw_error = evaluation.get("raw_error")
            if _is_valid_number(raw_error):
                raw_error = float(raw_error)
                cache_extra["raw_error"] = raw_error
                raw_errors.append(raw_error)

            metric_cache.save_item_score("segments", sample_id, idx, score, **cache_extra)
            scores.append(score)

        sample_score = aggregate_metric_scores(
            "motion_smoothness",
            scores,
            config=config,
            default_strategy="vde_decay",
            sample=sample,
        )
        results[sample_id] = sample_score
        metric_cache.save_sample_score(
            sample_id,
            sample_score,
            score_mapping=score_mapping,
            tau=tau,
            score_version=score_version,
            segment_raw_errors=raw_errors,
        )
    return results
