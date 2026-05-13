"""Conditional longrange consistency with cached MLLM grouping."""

from copy import deepcopy

import torch.nn.functional as F
from tqdm import tqdm

from narrastream_bench.models.dino_encoder import DINOEncoder
from narrastream_bench.utils.aggregation import (
    aggregate_scores_with_explicit_weights,
    apply_aggregation_strategy,
    get_metric_aggregation_config,
    get_sample_segment_weights,
    get_segment_subset_weights,
    remap_similarity_score,
    shrink_weights_toward_uniform,
    transform_coverage_weights,
)
from narrastream_bench.utils.local_cache import build_metric_result_cache, split_cached_samples
from narrastream_bench.utils.mllm_utils import extract_entity_groups_cached
from narrastream_bench.utils.video_io import load_segments


def _ensure_metric_cache_version(config, metric_name, default_cache_version):
    config_copy = deepcopy(config or {})
    aggregation_cfg = config_copy.setdefault("aggregation", {})
    metrics_cfg = aggregation_cfg.setdefault("metrics", {})
    metric_cfg = metrics_cfg.setdefault(metric_name, {})
    metric_cfg.setdefault("cache_version", default_cache_version)
    return config_copy


def compute_conditional_longrange(eval_data, device, config=None, path_config=None, **kwargs):
    cache_config = _ensure_metric_cache_version(
        config,
        "conditional_longrange",
        default_cache_version="v4_affine_power_weightshrink_raw",
    )
    metric_cache = build_metric_result_cache(
        "conditional_longrange",
        run_output_path=kwargs.get("run_output_path"),
        config=cache_config,
    )

    results, pending_samples = split_cached_samples(eval_data, metric_cache)
    if results:
        print(
            f"Conditional Longrange local cache hit: {len(results)}/{len(eval_data)} samples"
        )
    if not pending_samples:
        return results

    print("Loading DINO model for Longrange...")
    dino = DINOEncoder(device=device, config=config, path_config=path_config)
    agg_cfg = get_metric_aggregation_config("conditional_longrange", config)
    entity_strategy = agg_cfg.get("entity_strategy", "reverse_weighted")
    sample_strategy = agg_cfg.get("sample_strategy", "mean")
    fallback_entity_strategy = agg_cfg.get(
        "fallback_entity_strategy", "reverse_weighted"
    )
    fallback_sample_strategy = agg_cfg.get("fallback_sample_strategy", "mean")
    target_weight_alpha = max(0.0, min(1.0, float(agg_cfg.get("target_weight_alpha", 1.0))))
    sample_weight_alpha = max(0.0, min(1.0, float(agg_cfg.get("sample_weight_alpha", 1.0))))
    sample_weight_transform = str(agg_cfg.get("sample_weight_transform", "identity"))

    for sample in tqdm(pending_samples, desc="Conditional Longrange"):
        sample_id = sample["sample_id"]
        prompts = sample["prompts"]
        segments = load_segments(sample["segment_paths"], path_config=path_config)

        entity_groups = extract_entity_groups_cached(prompts, config=config)
        prompt_segment_weights = get_sample_segment_weights(sample=sample, config=config)

        all_scores = []
        sample_weights = []
        entity_records = []
        for entity, indices in entity_groups.items():
            if len(indices) < 2:
                continue

            anchor_index = int(indices[0])
            target_indices = [int(idx) for idx in indices[1:]]
            first_feat = dino.encode_segment(segments[anchor_index])
            target_weights = get_segment_subset_weights(prompt_segment_weights, target_indices)
            raw_similarities = []
            mapped_similarities = []

            for idx in target_indices:
                feat = dino.encode_segment(segments[idx])
                raw_similarity = F.cosine_similarity(first_feat, feat, dim=-1).item()
                mapped_similarity = remap_similarity_score(
                    raw_similarity,
                    metric_name="conditional_longrange",
                    config=config,
                )
                raw_similarities.append(float(raw_similarity))
                mapped_similarities.append(float(mapped_similarity))

            effective_target_weights = target_weights
            if (
                entity_strategy == "llm_prompt_target_weighted_mean"
                and target_weights is not None
            ):
                effective_target_weights = shrink_weights_toward_uniform(
                    target_weights,
                    alpha=target_weight_alpha,
                )

            if entity_strategy == "llm_prompt_target_weighted_mean":
                entity_score = aggregate_scores_with_explicit_weights(
                    mapped_similarities,
                    effective_target_weights,
                    fallback_strategy=fallback_entity_strategy,
                )
            else:
                entity_score = apply_aggregation_strategy(
                    mapped_similarities,
                    entity_strategy,
                )

            entity_sample_weight = None
            if prompt_segment_weights is not None:
                entity_sample_weight = float(
                    sum(
                        prompt_segment_weights[idx]
                        for idx in indices
                        if idx < len(prompt_segment_weights)
                    )
                )
                sample_weights.append(entity_sample_weight)

            entity_records.append(
                {
                    "entity": entity,
                    "indices": [int(idx) for idx in indices],
                    "anchor_index": anchor_index,
                    "target_indices": target_indices,
                    "raw_similarities": raw_similarities,
                    "mapped_similarities": mapped_similarities,
                    "target_weights": target_weights,
                    "effective_target_weights": effective_target_weights,
                    "sample_weight": entity_sample_weight,
                    "entity_score": float(entity_score),
                }
            )
            all_scores.append(entity_score)

        transformed_sample_weights = None
        effective_sample_weights = None
        if not all_scores:
            sample_score = None
        elif sample_strategy == "llm_prompt_coverage_weighted_mean":
            transformed_sample_weights = transform_coverage_weights(
                sample_weights,
                mode=sample_weight_transform,
            )
            effective_sample_weights = shrink_weights_toward_uniform(
                transformed_sample_weights,
                alpha=sample_weight_alpha,
            )
            sample_score = aggregate_scores_with_explicit_weights(
                all_scores,
                effective_sample_weights,
                fallback_strategy=fallback_sample_strategy,
            )
        else:
            sample_score = apply_aggregation_strategy(all_scores, sample_strategy)

        results[sample_id] = sample_score
        metric_cache.save_sample_state(
            sample_id,
            {
                "score": sample_score,
                "raw_metric_data": {
                    "entity_groups": {
                        entity: [int(idx) for idx in indices]
                        for entity, indices in entity_groups.items()
                    },
                    "entity_records": entity_records,
                    "prompt_segment_weights": prompt_segment_weights,
                    "sample_weights": sample_weights,
                    "transformed_sample_weights": transformed_sample_weights,
                    "effective_sample_weights": effective_sample_weights,
                    "entity_strategy": entity_strategy,
                    "sample_strategy": sample_strategy,
                    "fallback_entity_strategy": fallback_entity_strategy,
                    "fallback_sample_strategy": fallback_sample_strategy,
                    "target_weight_alpha": target_weight_alpha,
                    "sample_weight_alpha": sample_weight_alpha,
                    "sample_weight_transform": sample_weight_transform,
                },
            },
        )
    return results
