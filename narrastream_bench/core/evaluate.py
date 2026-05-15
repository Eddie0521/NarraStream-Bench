#!/usr/bin/env python
"""评估入口"""
import argparse
import json
import sys
import torch
from narrastream_bench.core.evaluator import NarraStreamBench


def _cuda_arch_is_supported(device_index=0):
    major, minor = torch.cuda.get_device_capability(device_index)
    required_arch = f"sm_{major}{minor}"
    supported_arches = set(torch.cuda.get_arch_list())
    return required_arch in supported_arches, required_arch, sorted(supported_arches)


def resolve_device(requested):
    """Resolve the requested torch device, guarding against unsupported CUDA builds."""
    requested = (requested or "auto").lower()
    if requested == "auto":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        supported, required_arch, supported_arches = _cuda_arch_is_supported()
        if supported:
            return torch.device("cuda")
        print(
            "Warning: CUDA device is visible, but this PyTorch build does not "
            f"support the GPU architecture {required_arch}. Falling back to CPU. "
            f"Supported architectures: {', '.join(supported_arches) or 'unknown'}",
            file=sys.stderr,
            flush=True,
        )
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type != "cuda":
        return device

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    device_index = 0 if device.index is None else device.index
    supported, required_arch, supported_arches = _cuda_arch_is_supported(device_index)
    if not supported:
        raise RuntimeError(
            "CUDA was requested, but this PyTorch build does not support "
            f"GPU architecture {required_arch}. Supported architectures: "
            f"{', '.join(supported_arches) or 'unknown'}. Install a PyTorch build "
            "with support for this GPU, or run with --device cpu."
        )
    return device


def main():
    parser = argparse.ArgumentParser(description='NarraStream-Bench Evaluation')
    parser.add_argument('--eval_data', required=True, help='eval_data.json 路径')
    parser.add_argument('--output', required=True, help='结果输出目录')
    parser.add_argument('--config', default=None, help='配置文件路径')
    parser.add_argument('--path_config', default=None, help='模型路径配置')
    parser.add_argument('--metrics', nargs='+', default=None, help='指标列表，默认全部')
    parser.add_argument('--vlm_model', default=None, help='覆盖配置中的 VLM 模型')
    parser.add_argument('--api-workers', type=int, default=4, help='API 指标并发 worker 数，默认 4')
    parser.add_argument(
        '--device',
        default='auto',
        help='Torch device: auto, cpu, cuda, or cuda:N. Default: auto',
    )
    parser.add_argument('--resume', action='store_true', help='从 output/results_latest.json 继续')
    parser.add_argument('--snapshot', default=None, help='快照文件路径，默认 output/results_latest.json')
    parser.add_argument(
        '--no-step-snapshots',
        action='store_true',
        help='关闭每个 metric 单独的快照文件输出',
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using torch device: {device}", flush=True)

    bench = NarraStreamBench(
        device=device,
        output_path=args.output,
        config_path=args.config,
        path_config=args.path_config
    )

    metrics = None if args.metrics == ['all'] else args.metrics

    results = bench.evaluate(
        args.eval_data,
        metric_list=metrics,
        vlm_model=args.vlm_model,
        api_workers=args.api_workers,
        resume=args.resume,
        snapshot_path=args.snapshot,
        save_step_snapshots=not args.no_step_snapshots,
    )

    print("=== Evaluation Results ===")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
