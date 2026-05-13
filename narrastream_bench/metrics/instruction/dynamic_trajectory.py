"""动态轨迹选择性：文本变化小时偏向稳定，文本变化大时要求响应。"""
from copy import deepcopy
import math

import torch.nn.functional as F

from narrastream_bench.models.languagebind_encoder import LanguageBindEncoder
from narrastream_bench.utils.aggregation import get_metric_aggregation_config, get_transition_subset_weights
from narrastream_bench.utils.local_cache import build_metric_result_cache, split_cached_samples
from tqdm import tqdm


def _change_magnitude(feat_a, feat_b):
    feat_a = F.normalize(feat_a, dim=-1)
    feat_b = F.normalize(feat_b, dim=-1)
    return max(0.0, 1.0 - F.cosine_similarity(feat_a, feat_b, dim=-1).item())


def _coerce_weights(weights, expected_length):
    if weights is None or len(weights) != expected_length:
        return [1.0] * expected_length

    coerced = [max(0.0, float(weight)) for weight in weights]
    if sum(coerced) <= 0.0:
        return [1.0] * expected_length
    return coerced


def _weighted_mean(values, weights=None):
    if not values:
        return 0.0

    weight_list = _coerce_weights(weights, len(values))
    weighted_sum = sum(value * weight for value, weight in zip(values, weight_list))
    return weighted_sum / sum(weight_list)


def _soft_gate(text_change, *, center, temp):
    scaled = (float(text_change) - float(center)) / max(1e-6, float(temp))
    scaled = max(-60.0, min(60.0, scaled))
    return 1.0 / (1.0 + math.exp(-scaled))


def _transition_selectivity_score(video_change, *, gate, tau_on, tau_off):
    video_change = max(0.0, float(video_change))
    response_score = 1.0 - math.exp(-video_change / max(1e-6, float(tau_on)))
    stability_score = math.exp(-video_change / max(1e-6, float(tau_off)))
    transition_score = float(gate) * response_score + (1.0 - float(gate)) * stability_score
    return transition_score, response_score, stability_score


def _ensure_metric_cache_version(config, metric_name, default_cache_version):
    config_copy = deepcopy(config or {})
    aggregation_cfg = config_copy.setdefault("aggregation", {})
    metrics_cfg = aggregation_cfg.setdefault("metrics", {})
    metric_cfg = metrics_cfg.setdefault(metric_name, {})
    metric_cfg.setdefault("cache_version", default_cache_version)
    return config_copy


def _build_transition_records(
    text_changes,
    video_changes,
    transition_scores,
    transition_gates,
    response_scores,
    stability_scores,
    transition_weights,
):
    records = []
    for index, (
        text_change,
        video_change,
        transition_score,
        transition_gate,
        response_score,
        stability_score,
    ) in enumerate(
        zip(
            text_changes,
            video_changes,
            transition_scores,
            transition_gates,
            response_scores,
            stability_scores,
        )
    ):
        records.append(
            {
                "transition_index": index,
                "text_change": float(text_change),
                "video_change": float(video_change),
                "transition_score": float(transition_score),
                "gate": float(transition_gate),
                "response_score": float(response_score),
                "stability_score": float(stability_score),
                "weight": (
                    None
                    if transition_weights is None or index >= len(transition_weights)
                    else float(transition_weights[index])
                ),
            }
        )
    return records


def compute_dynamic_trajectory(eval_data, device, config=None, path_config=None, **kwargs):
    cache_config = _ensure_metric_cache_version(
        config,
        "dynamic_trajectory",
        default_cache_version="transition_selectivity_v1_raw",
    )
    metric_cache = build_metric_result_cache(
        "dynamic_trajectory",
        run_output_path=kwargs.get("run_output_path"),
        config=cache_config,
    )
    metric_cfg = get_metric_aggregation_config("dynamic_trajectory", config)
    center = float(metric_cfg.get("center", 0.25))
    temp = float(metric_cfg.get("temp", 0.05))
    tau_on = float(metric_cfg.get("tau_on", 0.02))
    tau_off = float(metric_cfg.get("tau_off", 0.06))

    results, pending_samples = split_cached_samples(eval_data, metric_cache)
    if results:
        print(
            f"Dynamic Trajectory local cache hit: {len(results)}/{len(eval_data)} samples"
        )
    if not pending_samples:
        return results

    print("Loading LanguageBind model...")
    encoder = LanguageBindEncoder(device=device, config=config, path_config=path_config)

    for sample in tqdm(pending_samples, desc="Dynamic Trajectory"):
        sample_id = sample["sample_id"]
        segment_paths = sample['segment_paths']
        prompts = sample['prompts']
        
        v_feats = []
        p_feats = []
        
        for path, prompt in zip(segment_paths, prompts):
            v_feats.append(encoder.encode_video(path))
            p_feats.append(encoder.encode_text(prompt))

        transition_weights = get_transition_subset_weights(sample=sample, config=config)
        text_changes = []
        video_changes = []
        transition_scores = []
        transition_gates = []
        response_scores = []
        stability_scores = []
        for i in range(len(v_feats) - 1):
            text_change = _change_magnitude(p_feats[i], p_feats[i + 1])
            video_change = _change_magnitude(v_feats[i], v_feats[i + 1])
            gate = _soft_gate(
                text_change,
                center=center,
                temp=temp,
            )
            transition_score, response_score, stability_score = _transition_selectivity_score(
                video_change,
                gate=gate,
                tau_on=tau_on,
                tau_off=tau_off,
            )
            text_changes.append(text_change)
            video_changes.append(video_change)
            transition_scores.append(transition_score)
            transition_gates.append(gate)
            response_scores.append(response_score)
            stability_scores.append(stability_score)

        if not transition_scores:
            results[sample_id] = None
            metric_cache.save_sample_state(
                sample_id,
                {
                    "score": None,
                    "raw_metric_data": {
                        "transition_records": _build_transition_records(
                            text_changes,
                            video_changes,
                            transition_scores,
                            transition_gates,
                            response_scores,
                            stability_scores,
                            transition_weights,
                        ),
                        "center": center,
                        "temp": temp,
                        "tau_on": tau_on,
                        "tau_off": tau_off,
                        "transition_count": 0,
                    },
                },
            )
            continue

        sample_score = _weighted_mean(transition_scores, transition_weights)
        results[sample_id] = sample_score
        metric_cache.save_sample_state(
            sample_id,
            {
                "score": sample_score,
                "raw_metric_data": {
                    "transition_records": _build_transition_records(
                        text_changes,
                        video_changes,
                        transition_scores,
                        transition_gates,
                        response_scores,
                        stability_scores,
                        transition_weights,
                    ),
                    "center": center,
                    "temp": temp,
                    "tau_on": tau_on,
                    "tau_off": tau_off,
                    "transition_count": len(transition_scores),
                    "mean_gate": float(_weighted_mean(transition_gates, transition_weights)),
                    "mean_response_score": float(_weighted_mean(response_scores, transition_weights)),
                    "mean_stability_score": float(_weighted_mean(stability_scores, transition_weights)),
                },
            },
        )
    return results
