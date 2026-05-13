"""数据预处理：解析 prompts + 分割视频"""
import os
import json
import re
import time

from tqdm import tqdm


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif"}


def _natural_sort_key(path):
    """Sort paths like sample_2 before sample_10."""
    name = os.path.basename(path)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


def _discover_video_files(video_dir):
    files = []
    for entry in os.listdir(video_dir):
        full_path = os.path.join(video_dir, entry)
        if not os.path.isfile(full_path):
            continue
        if os.path.splitext(entry)[1].lower() in VIDEO_EXTENSIONS:
            files.append(full_path)
    return sorted(files, key=_natural_sort_key)


def split_video_to_segments(video_path, output_dir, segment_duration=10):
    """将完整视频按时长分割为多段"""
    import cv2

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # fallback
    w, h = int(cap.get(3)), int(cap.get(4))
    frames_per_seg = int(fps * segment_duration)
    
    paths, idx = [], 0
    while True:
        frames = []
        for _ in range(frames_per_seg):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        if not frames:
            break
        
        path = os.path.join(output_dir, f"seg_{idx}.mp4")
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in frames:
            out.write(f)
        out.release()
        paths.append(path)
        idx += 1
    
    cap.release()
    return paths


def parse_prompts_file(path):
    """解析 prompts 文件 (jsonl 或 json)"""
    samples = []
    if path.endswith('.jsonl'):
        with open(path) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    else:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            samples = data
        else:
            samples = [data]
    return samples


def prepare_evaluation_data(video_dir, prompts_file, output_dir, segment_duration=10):
    """准备评估数据"""
    os.makedirs(output_dir, exist_ok=True)
    samples = parse_prompts_file(prompts_file)
    eval_data = []
    total_samples = len(samples)

    print(
        "Preparing evaluation data: "
        f"samples={total_samples}, output_dir={output_dir}, segment_duration={segment_duration}s",
        flush=True,
    )

    sample_named_paths = [
        os.path.join(video_dir, f"sample_{idx}.mp4")
        for idx in range(len(samples))
    ]
    use_legacy_sample_names = any(os.path.exists(path) for path in sample_named_paths)

    if use_legacy_sample_names:
        video_paths = sample_named_paths
        print("Using legacy sample_{idx}.mp4 naming.", flush=True)
    else:
        video_paths = _discover_video_files(video_dir)
        if len(video_paths) != len(samples):
            raise ValueError(
                f"Found {len(video_paths)} video files in {video_dir}, "
                f"but prompts file contains {len(samples)} samples. "
                "Please make them match."
            )
        print(f"Using sorted video files from {video_dir}.", flush=True)

    progress = tqdm(
        total=total_samples,
        desc="Preprocess samples",
        unit="sample",
    )
    for idx, sample in enumerate(samples):
        video_path = video_paths[idx]
        sample_label = f"{idx + 1}/{total_samples}"
        video_name = os.path.basename(video_path)
        if not os.path.exists(video_path):
            tqdm.write(
                f"[preprocess] sample {sample_label} missing: video={video_name} path={video_path}",
            )
            progress.update(1)
            continue

        seg_dir = os.path.join(output_dir, f"sample_{idx}")
        tqdm.write(
            f"[preprocess] sample {sample_label} start: video={video_name} -> {seg_dir}",
        )
        started_at = time.monotonic()
        segment_paths = split_video_to_segments(video_path, seg_dir, segment_duration)
        elapsed = time.monotonic() - started_at
        tqdm.write(
            f"[preprocess] sample {sample_label} done: segments={len(segment_paths)}, elapsed={elapsed:.1f}s",
        )
        progress.set_postfix_str(
            f"last={video_name}, segments={len(segment_paths)}",
        )
        progress.update(1)

        eval_data.append({
            'sample_id': idx,
            'prompts': sample['prompts'],
            'segment_paths': segment_paths,
            'source_video': video_path,
        })
    progress.close()

    output_json = os.path.join(output_dir, 'eval_data.json')
    with open(output_json, 'w') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(eval_data)} samples -> {output_json}", flush=True)
    return eval_data
