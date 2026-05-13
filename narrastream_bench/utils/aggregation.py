"""聚合策略"""
import math

from narrastream_bench.utils.prompt_weight_planner import get_prompt_segment_weights


def mean_aggregation(scores):
    """简单均值"""
    return sum(scores) / len(scores) if scores else 0.0


def vde_decay(scores, weight_type='linear'):
    """VDE 漂移衰减：惩罚后续段分数下降"""
    if len(scores) < 2:
        return scores[0] if scores else 0.0
    
    baseline, n = scores[0], len(scores)
    weighted_sum, weight_sum = 0.0, 0.0
    
    for i, score in enumerate(scores[1:], start=2):
        delta = max(0, baseline - score) / (baseline + 1e-6)
        w = (n - i + 1) / n if weight_type == 'linear' else math.exp(-0.5 * (i-1))
        weighted_sum += w * delta
        weight_sum += w
    
    penalty = weighted_sum / weight_sum if weight_sum > 0 else 0
    return mean_aggregation(scores) * (1 - penalty)


def reverse_weighted(scores):
    """逆序加权：后面权重更大，惩罚长程不一致"""
    if not scores:
        return 0.0
    n = len(scores)
    weights = [(i + 1) / n for i in range(n)]
    return sum(w * s for w, s in zip(weights, scores)) / sum(weights)


def weighted_mean(scores, weights):
    """按显式权重做均值。"""
    if not scores:
        return 0.0
    if len(scores) != len(weights):
        raise ValueError("scores and weights must have the same length")
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("weights must sum to a positive value")
    return sum(score * weight for score, weight in zip(scores, weights)) / weight_sum


def get_llm_prompt_weighting_config(config=None):
    return ((config or {}).get("aggregation") or {}).get("llm_prompt_weighting", {})


def get_llm_prompt_fallback_strategy(config=None, default_strategy="mean"):
    fallback_strategy = get_llm_prompt_weighting_config(config).get(
        "fallback_strategy",
        default_strategy,
    )
    if str(fallback_strategy).startswith("llm_prompt_"):
        return default_strategy
    return fallback_strategy


def get_sample_segment_weights(sample=None, config=None):
    llm_cfg = get_llm_prompt_weighting_config(config)
    if llm_cfg.get("enabled", True) is False:
        return None

    prompts = (sample or {}).get("prompts")
    if not prompts:
        return None

    try:
        weights = get_prompt_segment_weights(prompts, config=config)
    except Exception:
        return None

    if len(weights) != len(prompts):
        return None
    return [float(weight) for weight in weights]


def get_segment_subset_weights(segment_weights, indices):
    if segment_weights is None:
        return None
    try:
        return [float(segment_weights[idx]) for idx in indices]
    except (IndexError, TypeError):
        return None


def normalize_nonnegative_weights(weights):
    if weights is None:
        return None
    cleaned = [max(0.0, float(weight)) for weight in weights]
    total = sum(cleaned)
    if total <= 0.0:
        return None
    return [weight / total for weight in cleaned]


def shrink_weights_toward_uniform(weights, alpha):
    normalized = normalize_nonnegative_weights(weights)
    if normalized is None:
        return None
    alpha = max(0.0, min(1.0, float(alpha)))
    uniform = 1.0 / len(normalized)
    return [
        alpha * weight + (1.0 - alpha) * uniform
        for weight in normalized
    ]


def transform_coverage_weights(weights, mode="identity"):
    if weights is None:
        return None
    cleaned = [max(0.0, float(weight)) for weight in weights]
    if mode == "identity":
        return cleaned
    if mode == "sqrt":
        return [math.sqrt(weight) for weight in cleaned]
    raise ValueError(f"Unsupported coverage weight transform: {mode}")


def project_transition_weights(segment_weights):
    if not segment_weights or len(segment_weights) < 2:
        return None
    return [
        float(segment_weights[i]) + float(segment_weights[i + 1])
        for i in range(len(segment_weights) - 1)
    ]


def get_transition_subset_weights(sample=None, config=None, indices=None):
    transition_weights = project_transition_weights(
        get_sample_segment_weights(sample=sample, config=config)
    )
    if transition_weights is None:
        return None
    if indices is None:
        return transition_weights
    return get_segment_subset_weights(transition_weights, indices)


def aggregate_scores_with_explicit_weights(
    scores,
    weights,
    *,
    fallback_strategy="mean",
    metric_name=None,
    sample=None,
    config=None,
):
    if scores and weights and len(scores) == len(weights):
        try:
            return weighted_mean(scores, weights)
        except (TypeError, ValueError):
            pass

    if str(fallback_strategy).startswith("llm_prompt_"):
        fallback_strategy = get_llm_prompt_fallback_strategy(
            config=config,
            default_strategy="mean",
        )

    return apply_aggregation_strategy(
        scores,
        fallback_strategy,
        metric_name=metric_name,
        sample=sample,
        config=config,
    )


def remap_similarity_score(score, *, metric_name=None, config=None):
    """Apply an optional post-processing curve to cosine-style similarities."""
    metric_cfg = get_metric_aggregation_config(metric_name, config)
    mapping = metric_cfg.get("similarity_mapping", "none")

    if mapping in {None, "none", "identity"}:
        return float(score)

    value = max(-1.0, min(1.0, float(score)))
    floor = float(metric_cfg.get("similarity_floor", 0.0))
    ceiling = float(metric_cfg.get("similarity_ceiling", 1.0))

    if ceiling <= floor:
        return 1.0 if value >= ceiling else 0.0

    normalized = (value - floor) / (ceiling - floor)
    normalized = max(0.0, min(1.0, normalized))

    if mapping == "affine_power":
        power = max(1e-6, float(metric_cfg.get("similarity_power", 1.0)))
        return normalized**power

    raise ValueError(f"Unsupported similarity mapping: {mapping}")


def llm_prompt_weighted_mean(metric_name, scores, sample=None, config=None):
    """Use an LLM-planned segment weighting derived from the prompt sequence."""
    if not scores:
        return 0.0

    llm_cfg = get_llm_prompt_weighting_config(config)
    if llm_cfg.get("enabled", True) is False:
        return mean_aggregation(scores)

    prompts = (sample or {}).get("prompts")
    fallback_strategy = get_llm_prompt_fallback_strategy(
        config=config,
        default_strategy="mean",
    )

    if not prompts or len(prompts) != len(scores):
        return apply_aggregation_strategy(
            scores,
            fallback_strategy,
            metric_name=metric_name,
            sample=sample,
            config=config,
        )

    try:
        weights = get_prompt_segment_weights(prompts, config=config)
        if len(weights) != len(scores):
            raise ValueError("weights length does not match scores length")
        return weighted_mean(scores, weights)
    except Exception:
        return apply_aggregation_strategy(
            scores,
            fallback_strategy,
            metric_name=metric_name,
            sample=sample,
            config=config,
        )


def get_metric_aggregation_config(metric_name, config=None):
    aggregation = (config or {}).get("aggregation", {})
    metrics = aggregation.get("metrics", {})
    return metrics.get(metric_name, {})


def apply_aggregation_strategy(
    scores,
    strategy,
    weight_type="linear",
    metric_name=None,
    sample=None,
    config=None,
):
    if strategy == "mean":
        return mean_aggregation(scores)
    if strategy == "vde_decay":
        return vde_decay(scores, weight_type=weight_type)
    if strategy == "reverse_weighted":
        return reverse_weighted(scores)
    if strategy == "llm_prompt_weighted_mean":
        return llm_prompt_weighted_mean(
            metric_name=metric_name,
            scores=scores,
            sample=sample,
            config=config,
        )
    if strategy in {"direct", "none"}:
        return scores[0] if scores else 0.0
    raise ValueError(f"Unsupported aggregation strategy: {strategy}")


def aggregate_metric_scores(
    metric_name,
    scores,
    config=None,
    default_strategy="mean",
    sample=None,
):
    metric_cfg = get_metric_aggregation_config(metric_name, config)
    strategy = metric_cfg.get("strategy", default_strategy)
    weight_type = metric_cfg.get("weight_type", "linear")
    return apply_aggregation_strategy(
        scores,
        strategy,
        weight_type=weight_type,
        metric_name=metric_name,
        sample=sample,
        config=config,
    )
