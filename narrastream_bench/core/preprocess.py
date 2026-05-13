#!/usr/bin/env python
"""数据预处理入口"""
import argparse
from narrastream_bench.utils.preprocessing import prepare_evaluation_data


def main():
    parser = argparse.ArgumentParser(description='NarraStream-Bench Preprocessing')
    parser.add_argument('--video_dir', required=True, help='视频目录')
    parser.add_argument('--prompts', required=True, help='prompts 文件路径')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--segment_duration', type=int, default=10, help='每段时长(秒)')
    args = parser.parse_args()
    
    prepare_evaluation_data(args.video_dir, args.prompts, args.output, args.segment_duration)


if __name__ == '__main__':
    main()
