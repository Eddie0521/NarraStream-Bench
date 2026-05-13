#!/usr/bin/env python
"""评估入口"""
import argparse
import json
import torch
from narrastream_bench.core.evaluator import NarraStreamBench


def main():
    parser = argparse.ArgumentParser(description='NarraStream-Bench Evaluation')
    parser.add_argument('--eval_data', required=True, help='eval_data.json 路径')
    parser.add_argument('--output', required=True, help='结果输出目录')
    parser.add_argument('--config', default=None, help='配置文件路径')
    parser.add_argument('--path_config', default=None, help='模型路径配置')
    parser.add_argument('--metrics', nargs='+', default=None, help='指标列表，默认全部')
    parser.add_argument('--vlm_model', default=None, help='覆盖配置中的 VLM 模型')
    parser.add_argument('--api-workers', type=int, default=4, help='API 指标并发 worker 数，默认 4')
    parser.add_argument('--resume', action='store_true', help='从 output/results_latest.json 继续')
    parser.add_argument('--snapshot', default=None, help='快照文件路径，默认 output/results_latest.json')
    parser.add_argument(
        '--no-step-snapshots',
        action='store_true',
        help='关闭每个 metric 单独的快照文件输出',
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
