"""Conditional adjacent consistency with cached MLLM requests."""

from copy import deepcopy

import torch.nn.functional as F
from tqdm import tqdm

from narrastream_bench.models.dino_encoder import DINOEncoder
from narrastream_bench.utils.aggregation import (
    aggregate_metric_scores,
    aggregate_scores_with_explicit_weights,
    get_metric_aggregation_config,
    get_sample_segment_weights,
    get_transition_subset_weights,
    project_transition_weights,
    remap_similarity_score,
)
from narrastream_bench.utils.local_cache import build_metric_result_cache, split_cached_samples
from narrastream_bench.utils.mllm_utils import batch_judge_scene_changes
from narrastream_bench.utils.video_io import load_segments


def _ensure_metric_cache_version(config, metric_name, default_cache_version):
    config_copy = deepcopy(config or {})
    aggregation_cfg = config_copy.setdefault("aggregation", {})
    metrics_cfg = aggregation_cfg.setdefault("metrics", {})
    metric_cfg = metrics_cfg.setdefault(metric_name, {})
    metric_cfg.setdefault("cache_version", default_cache_version)
    return config_copy


def compute_conditional_adjacent(eval_data, device, config=None, path_config=None, **kwargs):
    cache_config = _ensure_metric_cache_version(
        config,
        "conditional_adjacent",
        default_cache_version="v3_affine_power_raw",
    )
    metric_cache = build_metric_result_cache(
        "conditional_adjacent",
        run_output_path=kwargs.get("run_output_path"),
        config=cache_config,
    )
    metric_cfg = get_metric_aggregation_config("conditional_adjacent", config)
    strategy = metric_cfg.get("strategy", "mean")
    fallback_strategy = metric_cfg.get("fallback_strategy", "mean")

    results, pending_samples = split_cached_samples(eval_data, metric_cache)
    if results:
        print(
            f"Conditional Adjacent local cache hit: {len(results)}/{len(eval_data)} samples"
        )
    if not pending_samples:
        return results

    print("Loading DINO model for Adjacent Consistency...")
    dino = DINOEncoder(device=device, config=config, path_config=path_config)

    for sample in tqdm(pending_samples, desc="Conditional Adjacent"):
        sample_id = sample["sample_id"]
        prompts = sample["prompts"]
        segments = load_segments(sample["segment_paths"], path_config=path_config)
        prompt_segment_weights = get_sample_segment_weights(sample=sample, config=config)
        transition_weights = project_transition_weights(prompt_segment_weights)

        scores = []
        selected_transition_indices = []
        transition_records = []
        keep_flags = batch_judge_scene_changes(prompts, config=config)
        for i, should_keep in enumerate(keep_flags):
            record = {
                "transition_index": i,
                "should_keep": bool(should_keep),
                "raw_similarity": None,
                "score": None,
                "weight": (
                    None
                    if transition_weights is None or i >= len(transition_weights)
                    else float(transition_weights[i])
                ),
            }
            if should_keep:
                feat_i = dino.encode_segment(segments[i])
                feat_i1 = dino.encode_segment(segments[i + 1])
                raw_similarity = F.cosine_similarity(feat_i, feat_i1, dim=-1).item()
                sim = remap_similarity_score(
                    raw_similarity,
                    metric_name="conditional_adjacent",
                    config=config,
                )
                record["raw_similarity"] = float(raw_similarity)
                record["score"] = float(sim)
                scores.append(sim)
                selected_transition_indices.append(i)
            transition_records.append(record)

        raw_metric_data = {
            "transition_records": transition_records,
            "selected_transition_indices": list(selected_transition_indices),
            "prompt_segment_weights": prompt_segment_weights,
            "transition_weights": transition_weights,
            "strategy": strategy,
            "fallback_strategy": fallback_strategy,
        }

        if not scores:
            results[sample_id] = None
            metric_cache.save_sample_state(
                sample_id,
                {
                    "score": None,
                    "raw_metric_data": raw_metric_data,
                },
            )
            continue

        weights = None
        if strategy == "llm_prompt_transition_weighted_mean":
            weights = get_transition_subset_weights(
                sample=sample,
                config=config,
                indices=selected_transition_indices,
            )
            sample_score = aggregate_scores_with_explicit_weights(
                scores,
                weights,
                fallback_strategy=fallback_strategy,
                metric_name="conditional_adjacent",
                sample=sample,
                config=config,
            )
        else:
            sample_score = aggregate_metric_scores(
                "conditional_adjacent",
                scores,
                config=config,
                default_strategy="mean",
                sample=sample,
            )

        if weights is not None:
            for transition_index, weight in zip(selected_transition_indices, weights):
                transition_records[transition_index]["weight"] = float(weight)

        results[sample_id] = sample_score
        metric_cache.save_sample_state(
            sample_id,
            {
                "score": sample_score,
                "raw_metric_data": raw_metric_data,
            },
        )
    return results
