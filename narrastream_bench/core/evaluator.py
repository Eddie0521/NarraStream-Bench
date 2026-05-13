"""NarraStream-Bench core evaluator."""
import importlib
import json
import numbers
import os
import traceback
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from tqdm import tqdm


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s"


def load_config(config_path=None):
    """加载配置文件"""
    if config_path is None:
        repo_dir = Path(__file__).resolve().parents[2]
        config_path = repo_dir / 'configs' / 'default.yaml'

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def load_path_config(path_config=None):
    """加载模型路径配置"""
    if path_config is None:
        repo_dir = Path(__file__).resolve().parents[2]
        path_config = repo_dir / 'configs' / 'paths.yaml'

    if os.path.exists(path_config):
        with open(path_config, 'r') as f:
            return yaml.safe_load(f)
    return {}


class NarraStreamBench:
    def __init__(self, device, output_path, config_path=None, path_config=None):
        self.device = device
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

        self.config = load_config(config_path)
        self.path_config = load_path_config(path_config)

        self.metric_folder_map = {
            "subject_consistency": "quality",
            "background_consistency": "quality",
            "temporal_flickering": "quality",
            "motion_smoothness": "quality",
            "vtss": "quality",
            "boundary_smoothness": "temporal",
            "conditional_adjacent": "temporal",
            "conditional_longrange": "temporal",
            "entity_grounding": "instruction",
            "dynamic_trajectory": "instruction",
            "vlm_score": "instruction",
        }

    @staticmethod
    def _is_valid_score(score):
        return isinstance(score, numbers.Real) and not isinstance(score, bool)

    def _validate_metrics(self, metrics):
        unknown = [metric for metric in metrics if metric not in self.metric_folder_map]
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}")

    def _resolve_metric_module(self, metric):
        folder = self.metric_folder_map[metric]
        module_path = f"narrastream_bench.metrics.{folder}.{metric}"
        return importlib.import_module(module_path), module_path

    def _compute_metric_scores(self, metric, eval_data, **kwargs):
        module, module_path = self._resolve_metric_module(metric)
        compute_func = getattr(module, f"compute_{metric}")
        scores = compute_func(
            eval_data=eval_data,
            device=self.device,
            config=self.config,
            path_config=self.path_config,
            run_output_path=self.output_path,
            **kwargs,
        )
        return scores, module_path

    def evaluate(
        self,
        eval_data_path,
        metric_list=None,
        *,
        resume=False,
        snapshot_path=None,
        save_step_snapshots=True,
        **kwargs,
    ):
        """评估所有 samples"""
        if resume or snapshot_path:
            return self._evaluate_with_snapshots(
                eval_data_path,
                metric_list=metric_list,
                resume=resume,
                snapshot_path=snapshot_path,
                save_step_snapshots=save_step_snapshots,
                **kwargs,
            )

        with open(eval_data_path, 'r') as f:
            eval_data = json.load(f)

        if metric_list is None:
            metric_list = list(self.metric_folder_map.keys())
        self._validate_metrics(metric_list)

        total_samples = len(eval_data)
        total_metrics = len(metric_list)
        print(
            "Evaluation plan: "
            f"samples={total_samples}, metrics={total_metrics}, output={self.output_path}",
            flush=True,
        )

        per_sample_results = {s['sample_id']: {} for s in eval_data}
        aggregated_results = {}
        metric_coverage = {}

        metric_progress = tqdm(metric_list, desc='Metrics', unit='metric')
        for metric in metric_progress:
            metric_progress.set_postfix_str(metric)
            started_at = time.monotonic()
            module, module_path = self._resolve_metric_module(metric)
            tqdm.write(f"[metric] start: {metric} ({module_path})")
            compute_func = getattr(module, f"compute_{metric}")
            metric_scores = compute_func(
                eval_data=eval_data,
                device=self.device,
                config=self.config,
                path_config=self.path_config,
                run_output_path=self.output_path,
                **kwargs,
            )

            valid_scores = []
            if metric_scores:
                for sid, score in metric_scores.items():
                    per_sample_results[sid][metric] = score
                    if self._is_valid_score(score):
                        valid_scores.append(float(score))
                aggregated_results[metric] = (
                    sum(valid_scores) / len(valid_scores)
                    if valid_scores
                    else None
                )
                metric_coverage[metric] = {
                    'valid_count': len(valid_scores),
                    'total_count': len(metric_scores),
                }
            else:
                aggregated_results[metric] = None
                metric_coverage[metric] = {
                    'valid_count': 0,
                    'total_count': 0,
                }

            elapsed = time.monotonic() - started_at
            aggregated_score = aggregated_results[metric]
            aggregated_text = (
                f"{aggregated_score:.4f}"
                if self._is_valid_score(aggregated_score)
                else 'n/a'
            )
            coverage = metric_coverage[metric]
            tqdm.write(
                f"[metric] done: {metric}, avg={aggregated_text}, "
                f"valid={coverage['valid_count']}/{coverage['total_count']}, "
                f"elapsed={elapsed:.1f}s"
            )

        final_aggregated = self._calculate_dimension_scores(aggregated_results)
        final_aggregated["metric_coverage"] = metric_coverage

        print(
            "Evaluation finished: "
            f"quality={final_aggregated.get('quality_score')}, "
            f"temporal={final_aggregated.get('temporal_score')}, "
            f"instruction={final_aggregated.get('instruction_score')}, "
            f"total={final_aggregated.get('total_score')}",
            flush=True,
        )

        self._save_results(final_aggregated, per_sample_results)
        return final_aggregated

    def _evaluate_with_snapshots(
        self,
        eval_data_path,
        metric_list=None,
        *,
        resume=False,
        snapshot_path=None,
        save_step_snapshots=True,
        **kwargs,
    ):
        with open(eval_data_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        metrics = (
            list(self.metric_folder_map.keys())
            if metric_list is None
            else list(metric_list)
        )
        self._validate_metrics(metrics)

        latest_path = snapshot_path or os.path.join(self.output_path, "results_latest.json")
        step_dir = os.path.join(self.output_path, "steps")
        if save_step_snapshots:
            os.makedirs(step_dir, exist_ok=True)

        state = self._initial_state(
            eval_data=eval_data,
            metrics=metrics,
            eval_data_path=eval_data_path,
        )
        if resume and os.path.exists(latest_path):
            state = self._load_state_from_snapshot(
                snapshot_path=latest_path,
                eval_data=eval_data,
                metrics=metrics,
                fallback_state=state,
            )
            print(
                f"Resuming from {latest_path}: "
                f"{len(state['completed_metrics'])}/{len(metrics)} metrics already completed."
            )

        pending_metrics = [
            metric for metric in metrics if metric not in state["completed_metrics"]
        ]
        print(
            f"Resumable evaluation: {len(eval_data)} samples, "
            f"{len(state['completed_metrics'])}/{len(metrics)} metrics done, "
            f"{len(pending_metrics)} remaining."
        )
        print(f"Latest snapshot path: {latest_path}")
        if save_step_snapshots:
            print(f"Per-metric snapshots: {step_dir}")

        try:
            self._save_snapshot(
                state=state,
                snapshot_path=latest_path,
                status="running",
                current_metric=None,
                metrics=metrics,
            )

            for index, metric in enumerate(metrics, start=1):
                if metric in state["completed_metrics"]:
                    print(f"[{index}/{len(metrics)}] Skipping completed metric: {metric}")
                    continue

                started_at = _utc_now_iso()
                metric_start = perf_counter()
                print(f"[{index}/{len(metrics)}] Starting metric: {metric} at {started_at}")
                state["metric_timings"][metric] = {
                    "status": "running",
                    "started_at": started_at,
                }
                self._save_snapshot(
                    state=state,
                    snapshot_path=latest_path,
                    status="running",
                    current_metric=metric,
                    metrics=metrics,
                )

                metric_scores, module_path = self._compute_metric_scores(
                    metric,
                    eval_data,
                    **kwargs,
                )
                print(f"[{index}/{len(metrics)}] Loaded module: {module_path}")

                valid_scores = []
                if metric_scores:
                    for sid, score in metric_scores.items():
                        state["per_sample_results"].setdefault(sid, {})
                        state["per_sample_results"][sid][metric] = score
                        if self._is_valid_score(score):
                            valid_scores.append(float(score))
                    aggregated_value = (
                        sum(valid_scores) / len(valid_scores)
                        if valid_scores
                        else None
                    )
                    total_count = len(metric_scores)
                else:
                    aggregated_value = None
                    total_count = 0

                state["aggregated_results"][metric] = aggregated_value
                state["metric_coverage"][metric] = {
                    "valid_count": len(valid_scores),
                    "total_count": total_count,
                }
                state["completed_metrics"].append(metric)

                finished_at = _utc_now_iso()
                duration_seconds = perf_counter() - metric_start
                state["metric_timings"][metric] = {
                    "status": "completed",
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_seconds": round(duration_seconds, 3),
                    "valid_count": len(valid_scores),
                    "total_count": total_count,
                    "aggregated_score": aggregated_value,
                }
                state["metric_errors"].pop(metric, None)

                snapshot_payload = self._save_snapshot(
                    state=state,
                    snapshot_path=latest_path,
                    status="running",
                    current_metric=None,
                    metrics=metrics,
                )

                if save_step_snapshots:
                    step_path = os.path.join(step_dir, f"{index:02d}_{metric}.json")
                    self._write_json(step_path, snapshot_payload)
                    print(f"[{index}/{len(metrics)}] Saved per-metric snapshot to {step_path}")

                print(
                    f"[{index}/{len(metrics)}] Completed {metric} in "
                    f"{_format_duration(duration_seconds)} | "
                    f"score={aggregated_value} | "
                    f"coverage={len(valid_scores)}/{total_count}"
                )

        except Exception as exc:
            metric_name = locals().get("metric")
            if metric_name is not None:
                state["metric_errors"][metric_name] = {
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                state["metric_timings"].setdefault(metric_name, {})
                state["metric_timings"][metric_name].update(
                    {
                        "status": "failed",
                        "finished_at": _utc_now_iso(),
                    }
                )
            self._save_snapshot(
                state=state,
                snapshot_path=latest_path,
                status="failed",
                current_metric=metric_name,
                metrics=metrics,
            )
            raise

        final_payload = self._save_snapshot(
            state=state,
            snapshot_path=latest_path,
            status="completed",
            current_metric=None,
            metrics=metrics,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = os.path.join(self.output_path, f"results_{timestamp}.json")
        self._write_json(final_path, final_payload)
        print(f"Final results saved to {final_path}")
        return final_payload["aggregated"]

    def _initial_state(self, *, eval_data, metrics, eval_data_path):
        return {
            "run_started_at": _utc_now_iso(),
            "eval_data_path": os.path.abspath(eval_data_path),
            "sample_order": [sample["sample_id"] for sample in eval_data],
            "completed_metrics": [],
            "aggregated_results": {},
            "metric_coverage": {},
            "metric_timings": {},
            "metric_errors": {},
            "per_sample_results": {sample["sample_id"]: {} for sample in eval_data},
            "metrics": list(metrics),
        }

    def _load_state_from_snapshot(self, *, snapshot_path, eval_data, metrics, fallback_state):
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return fallback_state

        progress = payload.get("progress", {})
        completed_metrics = [
            metric
            for metric in progress.get("completed_metrics", [])
            if metric in metrics
        ]
        per_sample_results = {
            sample["sample_id"]: {
                key: value
                for key, value in sample.items()
                if key != "sample_id"
            }
            for sample in payload.get("per_sample", [])
        }
        for sample in eval_data:
            per_sample_results.setdefault(sample["sample_id"], {})

        aggregated = payload.get("aggregated", {})
        aggregated_results = {
            metric: aggregated.get(metric)
            for metric in completed_metrics
            if metric in aggregated
        }
        metric_coverage = dict(aggregated.get("metric_coverage", {}))

        return {
            "run_started_at": progress.get("run_started_at", fallback_state["run_started_at"]),
            "eval_data_path": progress.get("eval_data_path", fallback_state["eval_data_path"]),
            "sample_order": fallback_state["sample_order"],
            "completed_metrics": completed_metrics,
            "aggregated_results": aggregated_results,
            "metric_coverage": metric_coverage,
            "metric_timings": dict(progress.get("metric_timings", {})),
            "metric_errors": dict(progress.get("metric_errors", {})),
            "per_sample_results": per_sample_results,
            "metrics": list(metrics),
        }

    def _build_snapshot_payload(self, *, state, status, current_metric, metrics):
        aggregated = self._calculate_dimension_scores(dict(state["aggregated_results"]))
        aggregated["metric_coverage"] = dict(state["metric_coverage"])

        per_sample = []
        for sample_id in state["sample_order"]:
            item = dict(state["per_sample_results"].get(sample_id, {}))
            item["sample_id"] = sample_id
            per_sample.append(item)

        completed_metrics = list(state["completed_metrics"])
        pending_metrics = [metric for metric in metrics if metric not in completed_metrics]

        return {
            "aggregated": aggregated,
            "per_sample": per_sample,
            "config": self.config,
            "progress": {
                "status": status,
                "run_started_at": state["run_started_at"],
                "updated_at": _utc_now_iso(),
                "eval_data_path": state["eval_data_path"],
                "current_metric": current_metric,
                "total_metrics": len(metrics),
                "completed_count": len(completed_metrics),
                "completed_metrics": completed_metrics,
                "pending_metrics": pending_metrics,
                "metric_timings": dict(state["metric_timings"]),
                "metric_errors": dict(state["metric_errors"]),
            },
        }

    def _save_snapshot(self, *, state, snapshot_path, status, current_metric, metrics):
        payload = self._build_snapshot_payload(
            state=state,
            status=status,
            current_metric=current_metric,
            metrics=metrics,
        )
        self._write_json(snapshot_path, payload)
        print(f"Updated latest snapshot: {snapshot_path}")
        return payload

    def _calculate_dimension_scores(self, aggregated):
        """计算 Quality/Temporal/Instruction 维度分"""
        quality_metrics = ['subject_consistency', 'background_consistency',
                          'temporal_flickering', 'motion_smoothness', 'vtss']
        temporal_metrics = ['boundary_smoothness', 'conditional_adjacent',
                           'conditional_longrange']
        instruction_metrics = ['entity_grounding', 'dynamic_trajectory', 'vlm_score']

        weights = self.config.get('weights', {})
        w_quality = weights.get('quality', 1.0)
        w_temporal = weights.get('temporal', 1.0)
        w_instruction = weights.get('instruction', 1.0)

        def avg(metrics):
            vals = [
                aggregated[m]
                for m in metrics
                if m in aggregated and self._is_valid_score(aggregated[m])
            ]
            return sum(vals) / len(vals) if vals else None

        aggregated['quality_score'] = avg(quality_metrics)
        aggregated['temporal_score'] = avg(temporal_metrics)
        aggregated['instruction_score'] = avg(instruction_metrics)

        dimension_pairs = [
            (aggregated['quality_score'], w_quality),
            (aggregated['temporal_score'], w_temporal),
            (aggregated['instruction_score'], w_instruction),
        ]
        valid_dimensions = [
            (score, weight)
            for score, weight in dimension_pairs
            if self._is_valid_score(score) and weight > 0
        ]
        if valid_dimensions:
            weighted_sum = sum(score * weight for score, weight in valid_dimensions)
            total_weight = sum(weight for _, weight in valid_dimensions)
            aggregated['total_score'] = weighted_sum / total_weight
        else:
            aggregated['total_score'] = None

        return aggregated

    def _save_results(self, aggregated, per_sample):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        per_sample_list = []
        for sid, metrics in per_sample.items():
            item = metrics.copy()
            item['sample_id'] = sid
            per_sample_list.append(item)

        output = {
            'aggregated': aggregated,
            'per_sample': per_sample_list,
            'config': self.config
        }
        path = os.path.join(self.output_path, f'results_{timestamp}.json')
        self._write_json(path, output)
        print(f"Results saved to {path}")

    def _write_json(self, path: str, payload: dict[str, Any]) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
