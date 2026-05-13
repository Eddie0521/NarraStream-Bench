"""视频读写和加载工具"""
import os
import subprocess

from narrastream_bench.utils.video_primitives import load_video_frames


def load_segments(segment_paths, path_config=None):
    """加载视频分段为帧列表"""
    del path_config
    return [load_video_frames(p) for p in segment_paths]


def merge_segments(segment_paths, output_path):
    """合并分段视频为完整视频"""
    list_file = output_path.replace('.mp4', '_list.txt')
    with open(list_file, 'w') as f:
        for p in segment_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    
    subprocess.run([
        'ffmpeg', '-f', 'concat', '-safe', '0',
        '-i', list_file, '-c', 'copy', output_path, '-y'
    ], capture_output=True)
    
    if os.path.exists(list_file):
        os.remove(list_file)
    return output_path
