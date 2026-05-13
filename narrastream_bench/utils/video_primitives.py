"""Internal video loading and transform helpers."""
from pathlib import Path

import cv2
from PIL import Image, ImageSequence
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def build_dino_transform(size=224):
    """Match the DINO preprocessing expected by NarraStream-Bench metrics."""
    return Compose(
        [
            Resize(size=size),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def _sample_indices(total_frames, num_frames):
    if num_frames is None or num_frames <= 0 or total_frames <= num_frames:
        return list(range(total_frames))
    return sorted({int(round(i * (total_frames - 1) / (num_frames - 1))) for i in range(num_frames)})


def _load_gif_frames(path: Path):
    frames = []
    with Image.open(path) as image:
        for frame in ImageSequence.Iterator(image):
            frames.append(frame.convert("RGB").copy())
    return frames


def _load_image_frame(path: Path):
    with Image.open(path) as image:
        return [image.convert("RGB").copy()]


def _load_video_file(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")

    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))

    capture.release()

    if not frames:
        raise ValueError(f"No frames extracted from video: {path}")
    return frames


def load_video_frames(video_path, num_frames=None):
    """Load a video or image into a list of RGB PIL frames."""
    path = Path(video_path)
    suffix = path.suffix.lower()

    if suffix == ".gif":
        frames = _load_gif_frames(path)
    elif suffix in IMAGE_EXTENSIONS:
        frames = _load_image_frame(path)
    elif suffix in VIDEO_EXTENSIONS:
        frames = _load_video_file(path)
    else:
        raise NotImplementedError(f"Unsupported media type: {path}")

    indices = _sample_indices(len(frames), num_frames)
    return [frames[index] for index in indices]
