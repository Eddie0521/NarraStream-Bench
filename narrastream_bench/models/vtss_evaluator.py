"""VTSS runtime wrapper for NarraStream-Bench."""
from pathlib import Path

from narrastream_bench.third_party.vtss.vtss import VTSSCalculator
from narrastream_bench.utils.runtime_dependencies import resolve_repo_path


def _narrastream_bench_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_config_path() -> Path:
    return (
        _narrastream_bench_root()
        / "third_party"
        / "vtss"
        / "training_suitability_assessment"
        / "infer.yml"
    )


class VTSSEvaluator:
    def __init__(self, device="cuda", config=None, path_config=None):
        del config  # VTSS 主要通过 path_config 读取模型路径
        vtss_paths = (path_config or {}).get("vtss", {})

        self.config_path = resolve_repo_path(vtss_paths.get("config_path")) or _default_config_path()
        self.checkpoint_path = resolve_repo_path(vtss_paths.get("checkpoint_path"))

        self.calculator = VTSSCalculator(
            device,
            config_path=str(self.config_path),
            checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path else None,
        )

    def score_video(self, video_path: str) -> float | None:
        score = self.calculator.process_video(video_path)
        if score is None:
            return None
        return float(score)
