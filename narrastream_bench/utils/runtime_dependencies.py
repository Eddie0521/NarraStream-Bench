"""Helpers for local artifact loading."""
from pathlib import Path

import torch


NARRASTREAM_BENCH_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(raw_path):
    """Resolve a config path relative to the NarraStream-Bench repo root."""
    if not raw_path:
        return None

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (NARRASTREAM_BENCH_ROOT / path).resolve()

def load_torch_state_dict(weights_path):
    """Load a state dict from common checkpoint layouts."""
    checkpoint = torch.load(str(weights_path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ("state_dict", "model_state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            checkpoint = candidate
            break

    if checkpoint and all(isinstance(key, str) for key in checkpoint):
        checkpoint = {
            key.removeprefix("module."): value
            for key, value in checkpoint.items()
        }

    return checkpoint
