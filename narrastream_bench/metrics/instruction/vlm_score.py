"""VLM score metric with retry and frame caching."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local

from narrastream_bench.models.vlm_client import VLMClient
from tqdm import tqdm
from narrastream_bench.utils.aggregation import get_metric_aggregation_config
from narrastream_bench.utils.local_cache import build_metric_result_cache
from narrastream_bench.utils.api_retry import resolve_retry_policy, run_with_retry
from narrastream_bench.utils.vlm_score_utils import (
    build_vlm_prompt_text,
    parse_vlm_response_with_details,
    split_vlm_cached_samples,
)


_THREAD_LOCAL = local()


def _frame_cache_stats():
    cache_stats = getattr(VLMClient, "cache_stats", None)
    if callable(cache_stats):
        return cache_stats()
    return {"entries": 0, "hits": 0, "misses": 0}


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


def _score_sample(
    sample,
    *,
    config,
    path_config,
    vlm_model,
    segment_weight,
    overall_weight,
    fallback_score,
):
    vlm = _get_thread_local_vlm(vlm_model, config, path_config)
    prompts = sample["prompts"]
    segment_paths = sample["segment_paths"]
    frames_per_segment = (
        (config or {}).get("evaluation", {}).get("vlm_frames_per_segment", 5)
    )

    prompt_text = build_vlm_prompt_text(prompts)
    retry_policy = resolve_retry_policy(
        config=config,
        service_name="vlm",
        default_max_attempts=12,
        default_initial_delay=2.0,
        default_max_delay=60.0,
        default_backoff=1.8,
    )

    def _request_and_parse():
        raw_score = vlm.evaluate_segments(
            segment_paths,
            prompt_text,
            frames_per_segment=frames_per_segment,
            response_format={"type": "json_object"},
        )
        sample_state = parse_vlm_response_with_details(
            raw_score,
            expected_segments=len(prompts),
            sample=sample,
            config=config,
            segment_weight=segment_weight,
            overall_weight=overall_weight,
            fallback_score=fallback_score,
        )
        sample_state.setdefault("frames_per_segment", frames_per_segment)
        sample_state.setdefault("num_segments", len(prompts))
        return sample_state

    sample_state = run_with_retry(
        _request_and_parse,
        label=f"vlm_score sample={sample['sample_id']}",
        policy=retry_policy,
        retryable=lambda exc: True,
    )
    return sample["sample_id"], sample_state


def compute_vlm_score(eval_data, device, config=None, path_config=None, **kwargs):
    del device
    api_workers = max(1, int(kwargs.get("api_workers", 4)))
    print(f"Initializing VLM path with api_workers={api_workers}...")
    metric_cache = build_metric_result_cache(
        "vlm_score",
        run_output_path=kwargs.get("run_output_path"),
        config=config,
    )
    metric_cfg = get_metric_aggregation_config("vlm_score", config)
    segment_weight = float(metric_cfg.get("segment_weight", 0.8))
    overall_weight = float(metric_cfg.get("overall_weight", 0.2))
    fallback_score = float(metric_cfg.get("fallback_score", 0.5))

    results, pending_samples, cache_stats = split_vlm_cached_samples(
        eval_data,
        metric_cache,
        config=config,
        segment_weight=segment_weight,
        overall_weight=overall_weight,
        fallback_score=fallback_score,
    )
    if results:
        print(f"VLM Score local cache hit: {len(results)}/{len(eval_data)} samples")
    if cache_stats["upgraded_samples"]:
        print(
            "VLM Score local cache upgraded "
            f"(new weighting): {cache_stats['upgraded_samples']} samples"
        )
    if not pending_samples:
        print(f"VLM Score frame cache summary: {_frame_cache_stats()}")
        return results

    with ThreadPoolExecutor(max_workers=api_workers) as executor:
        future_to_sample = {
            executor.submit(
                _score_sample,
                sample,
                config=config,
                path_config=path_config,
                vlm_model=kwargs.get("vlm_model"),
                segment_weight=segment_weight,
                overall_weight=overall_weight,
                fallback_score=fallback_score,
            ): sample
            for sample in pending_samples
        }
        for future in tqdm(
            as_completed(future_to_sample),
            total=len(future_to_sample),
            desc="VLM Score",
        ):
            sample = future_to_sample[future]
            sample_id = sample["sample_id"]
            try:
                sample_id, sample_state = future.result()
            except Exception as exc:
                print(f"VLM Score failed for sample {sample_id}: {exc}")
                sample_state = {
                    "score": fallback_score,
                    "parse_mode": "request_failed",
                    "frames_per_segment": (
                        (config or {}).get("evaluation", {}).get("vlm_frames_per_segment", 5)
                    ),
                    "num_segments": len(sample.get("prompts", [])),
                    "error": str(exc),
                    "retry_exhausted": True,
                }
            results[sample_id] = sample_state["score"]
            cache_extra = {k: v for k, v in sample_state.items() if k != "score"}
            metric_cache.save_sample_score(sample_id, sample_state["score"], **cache_extra)

    print(f"VLM Score frame cache summary: {_frame_cache_stats()}")
    return results
