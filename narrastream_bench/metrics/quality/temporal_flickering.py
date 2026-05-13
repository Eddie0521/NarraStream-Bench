"""时序闪烁：使用 RAFT 对齐后的亮度残差评估帧间闪烁。"""
from __future__ import annotations

import json
import math
import numbers
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from narrastream_bench.models.raft_flow import RAFTFlow
from tqdm import tqdm

from narrastream_bench.utils.aggregation import aggregate_metric_scores
from narrastream_bench.utils.local_cache import build_metric_result_cache
from narrastream_bench.utils.video_primitives import load_video_frames


def _is_valid_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


_VALID_MASK_SUMMARY_PERCENTILES = (50.0, 75.0, 85.0, 90.0, 95.0)


def _percentile_key(percentile: float) -> str:
    if float(percentile).is_integer():
        return f"p{int(percentile)}"
    return f"p{str(percentile).replace('.', '_')}"


def _parse_percentile_reduction(name: str) -> float | None:
    reduction = str(name).strip().lower()
    if not reduction.startswith("p") or len(reduction) <= 1:
        return None
    try:
        percentile = float(reduction[1:])
    except ValueError:
        return None
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(
            "temporal_flickering percentile reduction must be in [0, 100], "
            f"got {name}"
        )
    return percentile


def _validate_reduction_name(
    name: str,
    *,
    field_name: str,
    allow_mean: bool = True,
    allow_max: bool = True,
) -> str:
    reduction = str(name).strip().lower()
    if allow_mean and reduction == "mean":
        return reduction
    if allow_max and reduction == "max":
        return reduction
    if _parse_percentile_reduction(reduction) is not None:
        return reduction
    allowed = []
    if allow_mean:
        allowed.append("mean")
    if allow_max:
        allowed.append("max")
    allowed.append("pXX")
    raise ValueError(
        f"Unsupported temporal_flickering {field_name}: {name}. "
        f"Expected one of {allowed}"
    )


def _metric_runtime_config(config=None) -> dict:
    metric_cfg = ((config or {}).get("aggregation") or {}).get("metrics", {}).get(
        "temporal_flickering",
        {},
    )
    raw_error_mode = str(metric_cfg.get("raw_error_mode", "raft_warped_luma"))
    frame_sample_count = int(metric_cfg.get("frame_sample_count", 8))
    fb_consistency_alpha = float(metric_cfg.get("fb_consistency_alpha", 0.01))
    fb_consistency_beta = float(metric_cfg.get("fb_consistency_beta", 0.5))
    min_mask_ratio = float(metric_cfg.get("min_mask_ratio", 0.05))
    mask_fallback = str(metric_cfg.get("mask_fallback", "in_bounds"))
    pair_spatial_reduction = _validate_reduction_name(
        metric_cfg.get("pair_spatial_reduction", "p90"),
        field_name="pair_spatial_reduction",
    )
    raw_error_reduction = _validate_reduction_name(
        metric_cfg.get("raw_error_reduction", "p84"),
        field_name="raw_error_reduction",
    )
    score_mapping = str(metric_cfg.get("score_mapping", "exp"))
    tau = float(metric_cfg.get("tau", 0.5))

    if raw_error_mode != "raft_warped_luma":
        raise ValueError(
            "Unsupported temporal_flickering raw_error_mode: "
            f"{raw_error_mode}"
        )
    if frame_sample_count < 2:
        raise ValueError(
            "temporal_flickering frame_sample_count must be at least 2, "
            f"got {frame_sample_count}"
        )
    if not 0.0 <= min_mask_ratio <= 1.0:
        raise ValueError(
            "temporal_flickering min_mask_ratio must be in [0, 1], "
            f"got {min_mask_ratio}"
        )
    if mask_fallback not in {"in_bounds", "skip"}:
        raise ValueError(
            "temporal_flickering mask_fallback must be one of "
            "{'in_bounds', 'skip'}, "
            f"got {mask_fallback}"
        )
    if score_mapping == "exp" and tau <= 0:
        raise ValueError(f"temporal_flickering tau must be positive, got {tau}")

    raw_error_version = (
        f"v4_{raw_error_mode}"
        f"_pair_{pair_spatial_reduction}"
        f"_frames_{frame_sample_count}"
        f"_alpha_{fb_consistency_alpha:.6f}"
        f"_beta_{fb_consistency_beta:.6f}"
        f"_minmask_{min_mask_ratio:.6f}"
        f"_fallback_{mask_fallback}"
    )
    score_version = (
        f"{raw_error_version}_{raw_error_reduction}_{score_mapping}_tau_{tau:.6f}"
    )
    return {
        "raw_error_mode": raw_error_mode,
        "frame_sample_count": frame_sample_count,
        "fb_consistency_alpha": fb_consistency_alpha,
        "fb_consistency_beta": fb_consistency_beta,
        "min_mask_ratio": min_mask_ratio,
        "mask_fallback": mask_fallback,
        "pair_spatial_reduction": pair_spatial_reduction,
        "raw_error_reduction": raw_error_reduction,
        "score_mapping": score_mapping,
        "tau": tau,
        "raw_error_version": raw_error_version,
        "score_version": score_version,
    }


def _frame_to_chw_float_tensor(frame) -> torch.Tensor:
    if isinstance(frame, torch.Tensor):
        tensor = frame.detach()
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 3:
            raise ValueError(f"Expected frame tensor with 3 dims, got {tensor.shape}")
        if tensor.shape[0] == 3:
            return tensor.to(dtype=torch.float32)
        if tensor.shape[-1] == 3:
            return tensor.permute(2, 0, 1).to(dtype=torch.float32)
        raise ValueError(f"Unsupported frame tensor shape: {tensor.shape}")

    array = np.asarray(frame, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Unsupported frame array shape: {array.shape}")
    return torch.from_numpy(array).permute(2, 0, 1)


def _frame_to_luma_tensor(frame_tensor: torch.Tensor) -> torch.Tensor:
    weights = frame_tensor.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    return (frame_tensor * weights).sum(dim=0, keepdim=True).unsqueeze(0)


def _warp_tensor_with_in_bounds(
    tensor: torch.Tensor,
    flow: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, height, width = tensor.shape
    if flow.shape != (batch, 2, height, width):
        raise ValueError(
            "warp flow shape mismatch: "
            f"tensor={tuple(tensor.shape)} flow={tuple(flow.shape)}"
        )

    x_coords = torch.arange(width, device=tensor.device, dtype=tensor.dtype)
    y_coords = torch.arange(height, device=tensor.device, dtype=tensor.dtype)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    xx = xx.unsqueeze(0).expand(batch, -1, -1)
    yy = yy.unsqueeze(0).expand(batch, -1, -1)

    sample_x = xx + flow[:, 0]
    sample_y = yy + flow[:, 1]
    in_bounds = (
        (sample_x >= 0.0)
        & (sample_x <= max(width - 1, 0))
        & (sample_y >= 0.0)
        & (sample_y <= max(height - 1, 0))
    )

    if width > 1:
        grid_x = 2.0 * sample_x / float(width - 1) - 1.0
    else:
        grid_x = torch.zeros_like(sample_x)
    if height > 1:
        grid_y = 2.0 * sample_y / float(height - 1) - 1.0
    else:
        grid_y = torch.zeros_like(sample_y)

    grid = torch.stack((grid_x, grid_y), dim=-1)
    warped = F.grid_sample(
        tensor,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped, in_bounds.unsqueeze(1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> float | None:
    if mask is None:
        return None
    selected = values.masked_select(mask)
    if selected.numel() <= 0:
        return None
    return float(selected.mean().item())


def _masked_percentile(
    values: torch.Tensor,
    mask: torch.Tensor | None,
    percentile: float,
) -> float | None:
    if mask is None:
        return None
    selected = values.masked_select(mask)
    if selected.numel() <= 0:
        return None
    return float(np.percentile(selected.detach().cpu().numpy(), percentile))


def _masked_array(
    values: torch.Tensor,
    mask: torch.Tensor | None,
) -> np.ndarray | None:
    if mask is None:
        return None
    selected = values.masked_select(mask)
    if selected.numel() <= 0:
        return None
    return selected.detach().cpu().numpy()


def _summarize_array(
    values: np.ndarray | None,
    *,
    percentiles: tuple[float, ...] = (),
) -> dict[str, float | None]:
    if values is None or values.size <= 0:
        summary: dict[str, float | None] = {"mean": None, "max": None}
        for percentile in percentiles:
            summary[_percentile_key(percentile)] = None
        return summary

    summary = {
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }
    for percentile in percentiles:
        summary[_percentile_key(percentile)] = float(np.percentile(values, percentile))
    return summary


def _reduce_array(
    values: np.ndarray | None,
    *,
    reduction: str,
    context_name: str,
) -> float | None:
    if values is None or values.size <= 0:
        return None

    if reduction == "mean":
        return float(np.mean(values))
    if reduction == "max":
        return float(np.max(values))

    percentile = _parse_percentile_reduction(reduction)
    if percentile is not None:
        return float(np.percentile(values, percentile))

    raise ValueError(
        f"Unsupported temporal_flickering {context_name}: {reduction}"
    )


def _compute_pair_metrics(
    frame_a: torch.Tensor,
    frame_b: torch.Tensor,
    luma_a: torch.Tensor,
    luma_b: torch.Tensor,
    raft: RAFTFlow,
    *,
    pair_index: int,
    fb_consistency_alpha: float,
    fb_consistency_beta: float,
    min_mask_ratio: float,
    mask_fallback: str,
    pair_spatial_reduction: str,
) -> dict:
    forward_flow = raft.compute_flow(frame_a, frame_b).unsqueeze(0)
    backward_flow = raft.compute_flow(frame_b, frame_a).unsqueeze(0)

    warped_luma_b, in_bounds_mask = _warp_tensor_with_in_bounds(luma_b, forward_flow)
    warped_backward_flow, _ = _warp_tensor_with_in_bounds(backward_flow, forward_flow)

    residual = torch.abs(luma_a - warped_luma_b) / 255.0
    consistency_sq = (forward_flow + warped_backward_flow).pow(2).sum(
        dim=1,
        keepdim=True,
    )
    forward_norm_sq = forward_flow.pow(2).sum(dim=1, keepdim=True)
    backward_norm_sq = warped_backward_flow.pow(2).sum(dim=1, keepdim=True)
    consistency_mask = in_bounds_mask & (
        consistency_sq
        < (
            fb_consistency_alpha * (forward_norm_sq + backward_norm_sq)
            + fb_consistency_beta
        )
    )

    in_bounds_ratio = float(in_bounds_mask.float().mean().item())
    consistency_mask_ratio = float(consistency_mask.float().mean().item())

    if consistency_mask_ratio >= min_mask_ratio:
        valid_mask = consistency_mask
        mask_mode = "fb_consistency"
    elif mask_fallback == "in_bounds" and in_bounds_ratio > 0.0:
        valid_mask = in_bounds_mask
        mask_mode = "in_bounds_fallback"
    else:
        valid_mask = None
        mask_mode = "skipped"

    valid_mask_ratio = (
        float(valid_mask.float().mean().item()) if valid_mask is not None else 0.0
    )
    consistency_error = torch.sqrt(consistency_sq + 1e-12)
    flow_magnitude = torch.sqrt(forward_norm_sq + 1e-12)
    backward_flow_magnitude = torch.sqrt(backward_norm_sq + 1e-12)
    residual_in_bounds_values = _masked_array(residual, in_bounds_mask)
    residual_valid_values = _masked_array(residual, valid_mask)
    residual_in_bounds_stats = _summarize_array(
        residual_in_bounds_values,
        percentiles=(90.0,),
    )
    residual_valid_stats = _summarize_array(
        residual_valid_values,
        percentiles=_VALID_MASK_SUMMARY_PERCENTILES,
    )
    consistency_error_stats = _summarize_array(
        _masked_array(consistency_error, in_bounds_mask),
        percentiles=(90.0,),
    )
    flow_magnitude_stats = _summarize_array(_masked_array(flow_magnitude, in_bounds_mask))
    backward_flow_magnitude_stats = _summarize_array(
        _masked_array(backward_flow_magnitude, in_bounds_mask)
    )
    pair_raw_error = _reduce_array(
        residual_valid_values,
        reduction=pair_spatial_reduction,
        context_name="pair_spatial_reduction",
    )

    record = {
        "pair_index": int(pair_index),
        "frame_index_a": int(pair_index),
        "frame_index_b": int(pair_index + 1),
        "mask_mode": mask_mode,
        "pair_spatial_reduction": pair_spatial_reduction,
        "raw_error": pair_raw_error,
        "in_bounds_ratio": in_bounds_ratio,
        "consistency_mask_ratio": consistency_mask_ratio,
        "valid_mask_ratio": valid_mask_ratio,
        "in_bounds_pixel_count": (
            int(residual_in_bounds_values.size)
            if residual_in_bounds_values is not None
            else 0
        ),
        "valid_pixel_count": (
            int(residual_valid_values.size) if residual_valid_values is not None else 0
        ),
        "residual_in_bounds_mean": residual_in_bounds_stats["mean"],
        "residual_in_bounds_p90": residual_in_bounds_stats["p90"],
        "residual_valid_mask_mean": residual_valid_stats["mean"],
        "residual_valid_mask_max": residual_valid_stats["max"],
        "residual_valid_mask_p50": residual_valid_stats["p50"],
        "residual_valid_mask_p75": residual_valid_stats["p75"],
        "residual_valid_mask_p85": residual_valid_stats["p85"],
        "residual_valid_mask_p90": residual_valid_stats["p90"],
        "residual_valid_mask_p95": residual_valid_stats["p95"],
        "fb_consistency_error_mean": consistency_error_stats["mean"],
        "fb_consistency_error_p90": consistency_error_stats["p90"],
        "flow_magnitude_mean": flow_magnitude_stats["mean"],
        "backward_flow_magnitude_mean": backward_flow_magnitude_stats["mean"],
    }
    if valid_mask is None:
        record["skip_reason"] = "insufficient_valid_mask"
    return record


def _compute_segment_intermediates(
    seg_frames,
    raft: RAFTFlow,
    runtime_cfg: dict,
) -> tuple[list[float], list[dict], int]:
    if len(seg_frames) < 2:
        return [], [], len(seg_frames)

    frame_tensors = [
        _frame_to_chw_float_tensor(frame).to(device=raft.device, dtype=torch.float32)
        for frame in seg_frames
    ]
    luma_tensors = [_frame_to_luma_tensor(frame_tensor) for frame_tensor in frame_tensors]

    pair_metrics = []
    with torch.no_grad():
        for pair_index in range(len(frame_tensors) - 1):
            pair_metrics.append(
                _compute_pair_metrics(
                    frame_tensors[pair_index],
                    frame_tensors[pair_index + 1],
                    luma_tensors[pair_index],
                    luma_tensors[pair_index + 1],
                    raft,
                    pair_index=pair_index,
                    fb_consistency_alpha=runtime_cfg["fb_consistency_alpha"],
                    fb_consistency_beta=runtime_cfg["fb_consistency_beta"],
                    min_mask_ratio=runtime_cfg["min_mask_ratio"],
                    mask_fallback=runtime_cfg["mask_fallback"],
                    pair_spatial_reduction=runtime_cfg["pair_spatial_reduction"],
                )
            )

    pair_raw_errors = [
        float(record["raw_error"])
        for record in pair_metrics
        if _is_valid_number(record.get("raw_error"))
    ]
    return pair_raw_errors, pair_metrics, len(frame_tensors)


def _build_segment_record(
    *,
    segment_index: int,
    raw_error: float,
    segment_score: float,
    pair_raw_errors: list[float],
    pair_metrics: list[dict],
    sampled_frame_count: int,
) -> dict:
    return {
        "segment_index": int(segment_index),
        "raw_error": float(raw_error),
        "segment_score": float(segment_score),
        "sampled_frame_count": int(sampled_frame_count),
        "pair_count": int(len(pair_metrics)),
        "valid_pair_count": int(len(pair_raw_errors)),
        "pair_raw_errors": [float(value) for value in pair_raw_errors],
        "pair_metrics": pair_metrics,
    }


def _build_sample_raw_metric_data(
    *,
    runtime_cfg: dict,
    strategy: str,
    segment_records: list[dict],
) -> dict:
    return {
        "raw_error_mode": runtime_cfg["raw_error_mode"],
        "frame_sample_count": int(runtime_cfg["frame_sample_count"]),
        "fb_consistency_alpha": float(runtime_cfg["fb_consistency_alpha"]),
        "fb_consistency_beta": float(runtime_cfg["fb_consistency_beta"]),
        "min_mask_ratio": float(runtime_cfg["min_mask_ratio"]),
        "mask_fallback": runtime_cfg["mask_fallback"],
        "pair_spatial_reduction": runtime_cfg["pair_spatial_reduction"],
        "raw_error_reduction": runtime_cfg["raw_error_reduction"],
        "score_mapping": runtime_cfg["score_mapping"],
        "tau": float(runtime_cfg["tau"]),
        "aggregation_strategy": strategy,
        "segment_records": segment_records,
    }


def _write_raw_metric_export(
    *,
    metric_cache,
    eval_data,
    run_output_path,
    score_version: str,
):
    if not run_output_path:
        return
    export_dir = Path(run_output_path) / "raw_metrics"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "temporal_flickering.jsonl"
    with export_path.open("w", encoding="utf-8") as handle:
        for sample in eval_data:
            sample_state = metric_cache.load_sample_state(sample["sample_id"])
            if not isinstance(sample_state, dict):
                continue
            if sample_state.get("score_version") != score_version:
                continue
            handle.write(json.dumps(sample_state, ensure_ascii=False))
            handle.write("\n")


def _reduce_raw_errors(raw_errors, *, raw_error_reduction: str) -> float | None:
    if not raw_errors:
        return None

    return _reduce_array(
        np.asarray(raw_errors, dtype=np.float32),
        reduction=raw_error_reduction,
        context_name="raw_error_reduction",
    )


def _score_from_raw_error(raw_error: float, *, score_mapping: str, tau: float) -> float:
    if score_mapping == "exp":
        return float(math.exp(-float(raw_error) / tau))
    if score_mapping == "linear":
        return max(0.0, 1.0 - float(raw_error))

    raise ValueError(
        f"Unsupported temporal_flickering score_mapping: {score_mapping}"
    )


def compute_temporal_flickering(eval_data, device, config=None, path_config=None, **kwargs):
    metric_cache = build_metric_result_cache(
        "temporal_flickering",
        run_output_path=kwargs.get("run_output_path"),
        config=config,
    )
    runtime_cfg = _metric_runtime_config(config=config)
    raw_error_reduction = runtime_cfg["raw_error_reduction"]
    score_mapping = runtime_cfg["score_mapping"]
    tau = runtime_cfg["tau"]
    raw_error_version = runtime_cfg["raw_error_version"]
    score_version = runtime_cfg["score_version"]
    strategy = (
        (((config or {}).get("aggregation") or {}).get("metrics") or {})
        .get("temporal_flickering", {})
        .get("strategy", "vde_decay")
    )

    results = {}
    pending_samples = []
    for sample in eval_data:
        sample_state = metric_cache.load_sample_state(sample["sample_id"])
        if (
            isinstance(sample_state, dict)
            and "score" in sample_state
            and sample_state.get("score_version") == score_version
            and sample_state.get("aggregation_strategy") == strategy
        ):
            results[sample["sample_id"]] = sample_state["score"]
            continue
        pending_samples.append(sample)

    if results:
        print(
            f"Temporal Flickering local cache hit: {len(results)}/{len(eval_data)} samples"
        )
    if not pending_samples:
        _write_raw_metric_export(
            metric_cache=metric_cache,
            eval_data=eval_data,
            run_output_path=kwargs.get("run_output_path"),
            score_version=score_version,
        )
        return results

    raft = None

    def get_raft():
        nonlocal raft
        if raft is None:
            print("Loading RAFT model for Temporal Flickering...")
            raft = RAFTFlow(device=device, config=config, path_config=path_config)
        return raft

    for sample in tqdm(pending_samples, desc="Temporal Flickering"):
        sample_id = sample["sample_id"]
        scores = []
        raw_errors = []
        segment_records = []

        for idx, segment_path in enumerate(sample["segment_paths"]):
            cached_item_state = metric_cache.load_item_state("segments", sample_id, idx)
            if isinstance(cached_item_state, dict):
                cached_pair_raw_errors = cached_item_state.get("pair_raw_errors")
                if (
                    cached_item_state.get("raw_error_version") == raw_error_version
                    and isinstance(cached_pair_raw_errors, list)
                ):
                    pair_raw_errors = [
                        float(value)
                        for value in cached_pair_raw_errors
                        if _is_valid_number(value)
                    ]
                    raw_error = _reduce_raw_errors(
                        pair_raw_errors,
                        raw_error_reduction=raw_error_reduction,
                    )
                    if raw_error is not None:
                        segment_score = _score_from_raw_error(
                            raw_error,
                            score_mapping=score_mapping,
                            tau=tau,
                        )
                        pair_metrics = cached_item_state.get("pair_metrics") or []
                        sampled_frame_count = int(
                            cached_item_state.get("sampled_frame_count", 0)
                        )
                        metric_cache.save_item_score(
                            "segments",
                            sample_id,
                            idx,
                            segment_score,
                            raw_error=raw_error,
                            raw_error_mode=runtime_cfg["raw_error_mode"],
                            frame_sample_count=runtime_cfg["frame_sample_count"],
                            fb_consistency_alpha=runtime_cfg["fb_consistency_alpha"],
                            fb_consistency_beta=runtime_cfg["fb_consistency_beta"],
                            min_mask_ratio=runtime_cfg["min_mask_ratio"],
                            mask_fallback=runtime_cfg["mask_fallback"],
                            pair_spatial_reduction=runtime_cfg["pair_spatial_reduction"],
                            raw_error_reduction=raw_error_reduction,
                            raw_error_version=raw_error_version,
                            pair_raw_errors=pair_raw_errors,
                            pair_metrics=pair_metrics,
                            sampled_frame_count=sampled_frame_count,
                            score_mapping=score_mapping,
                            tau=tau,
                            score_version=score_version,
                        )
                        scores.append(segment_score)
                        raw_errors.append(raw_error)
                        segment_records.append(
                            _build_segment_record(
                                segment_index=idx,
                                raw_error=raw_error,
                                segment_score=segment_score,
                                pair_raw_errors=pair_raw_errors,
                                pair_metrics=pair_metrics,
                                sampled_frame_count=sampled_frame_count,
                            )
                        )
                        continue

                cached_raw_error = cached_item_state.get("raw_error")
                cached_segment_score = cached_item_state.get("score")
                if (
                    cached_item_state.get("score_version") == score_version
                    and _is_valid_number(cached_segment_score)
                    and _is_valid_number(cached_raw_error)
                ):
                    raw_error = float(cached_raw_error)
                    segment_score = float(cached_segment_score)
                    pair_raw_errors = [
                        float(value)
                        for value in (cached_item_state.get("pair_raw_errors") or [])
                        if _is_valid_number(value)
                    ]
                    pair_metrics = cached_item_state.get("pair_metrics") or []
                    sampled_frame_count = int(
                        cached_item_state.get("sampled_frame_count", 0)
                    )
                    scores.append(segment_score)
                    raw_errors.append(raw_error)
                    segment_records.append(
                        _build_segment_record(
                            segment_index=idx,
                            raw_error=raw_error,
                            segment_score=segment_score,
                            pair_raw_errors=pair_raw_errors,
                            pair_metrics=pair_metrics,
                            sampled_frame_count=sampled_frame_count,
                        )
                    )
                    continue

            seg_frames = load_video_frames(
                segment_path,
                num_frames=runtime_cfg["frame_sample_count"],
            )
            pair_raw_errors, pair_metrics, sampled_frame_count = (
                _compute_segment_intermediates(
                    seg_frames,
                    get_raft(),
                    runtime_cfg,
                )
            )
            raw_error = _reduce_raw_errors(
                pair_raw_errors,
                raw_error_reduction=raw_error_reduction,
            )
            if raw_error is None:
                continue

            segment_score = _score_from_raw_error(
                raw_error,
                score_mapping=score_mapping,
                tau=tau,
            )
            metric_cache.save_item_score(
                "segments",
                sample_id,
                idx,
                segment_score,
                raw_error=raw_error,
                raw_error_mode=runtime_cfg["raw_error_mode"],
                frame_sample_count=runtime_cfg["frame_sample_count"],
                fb_consistency_alpha=runtime_cfg["fb_consistency_alpha"],
                fb_consistency_beta=runtime_cfg["fb_consistency_beta"],
                min_mask_ratio=runtime_cfg["min_mask_ratio"],
                mask_fallback=runtime_cfg["mask_fallback"],
                pair_spatial_reduction=runtime_cfg["pair_spatial_reduction"],
                raw_error_reduction=raw_error_reduction,
                raw_error_version=raw_error_version,
                pair_raw_errors=pair_raw_errors,
                pair_metrics=pair_metrics,
                sampled_frame_count=sampled_frame_count,
                score_mapping=score_mapping,
                tau=tau,
                score_version=score_version,
            )
            scores.append(segment_score)
            raw_errors.append(raw_error)
            segment_records.append(
                _build_segment_record(
                    segment_index=idx,
                    raw_error=raw_error,
                    segment_score=segment_score,
                    pair_raw_errors=pair_raw_errors,
                    pair_metrics=pair_metrics,
                    sampled_frame_count=sampled_frame_count,
                )
            )

        sample_score = None
        if scores:
            sample_score = aggregate_metric_scores(
                "temporal_flickering",
                scores,
                config=config,
                default_strategy="vde_decay",
                sample=sample,
            )
        results[sample_id] = sample_score
        metric_cache.save_sample_state(
            sample_id,
            {
                "score": sample_score,
                "raw_error_mode": runtime_cfg["raw_error_mode"],
                "frame_sample_count": runtime_cfg["frame_sample_count"],
                "fb_consistency_alpha": runtime_cfg["fb_consistency_alpha"],
                "fb_consistency_beta": runtime_cfg["fb_consistency_beta"],
                "min_mask_ratio": runtime_cfg["min_mask_ratio"],
                "mask_fallback": runtime_cfg["mask_fallback"],
                "pair_spatial_reduction": runtime_cfg["pair_spatial_reduction"],
                "raw_error_reduction": raw_error_reduction,
                "raw_error_version": raw_error_version,
                "score_mapping": score_mapping,
                "tau": tau,
                "aggregation_strategy": strategy,
                "score_version": score_version,
                "segment_raw_errors": raw_errors,
                "raw_metric_data": _build_sample_raw_metric_data(
                    runtime_cfg=runtime_cfg,
                    strategy=strategy,
                    segment_records=segment_records,
                ),
            },
        )

    _write_raw_metric_export(
        metric_cache=metric_cache,
        eval_data=eval_data,
        run_output_path=kwargs.get("run_output_path"),
        score_version=score_version,
    )
    return results
