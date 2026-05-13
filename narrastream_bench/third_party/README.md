# Third-party Code

This directory contains small subsets of external repositories vendored into
NarraStream-Bench so runtime no longer depends on local checkouts of those projects.
These components are used as model/runtime backbones. The final NarraStream-Bench
metrics, normalization, and cross-segment aggregation are implemented under
`metrics/` and `utils/`.

- `utils/video_primitives.py` includes video loading/preprocessing helpers
  adapted from VBench
  (`https://github.com/Vchitect/VBench`, commit `61dfa7d9136e35023edd1266a13de679b52fdd31`).
- `third_party/raft/` contains the RAFT runtime used for optical-flow
  features. The NarraStream-Bench flickering and boundary metrics are not direct
  outputs from an external benchmark.
- `third_party/amt/` contains the AMT-S runtime used for frame
  interpolation. NarraStream-Bench maps the interpolation error into its own
  `motion_smoothness` score.
- `third_party/vtss/` contains VTSS inference code adapted from
  IVEBench (`https://github.com/RyanChenYN/IVEBench`, commit `2dcf8900ee65197f43854c76f61c66a32bb8dbfc`).
  NarraStream-Bench uses the raw VTSS output, then applies its own normalization and
  segment aggregation.

External model checkpoints are not bundled with this repository. Use
`scripts/download_weights.sh` or set custom paths in `configs/paths.yaml`.
