"""背景一致性：段内 p25 + 跨段代表帧一致性。"""

import torch
import torch.nn.functional as F

from narrastream_bench.models.clip_encoder import ViCLIPEncoder
from narrastream_bench.utils.consistency_metric import compute_consistency_metric


def _encode_clip_frames(encoder, frames):
    if not frames:
        return []

    with torch.no_grad():
        images = torch.stack([encoder.preprocess(frame) for frame in frames]).to(encoder.device)
        image_features = encoder.model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1, p=2)
    return [feature for feature in image_features]


def compute_background_consistency(eval_data, device, config=None, path_config=None, **kwargs):
    return compute_consistency_metric(
        metric_name="background_consistency",
        eval_data=eval_data,
        device=device,
        config=config,
        path_config=path_config,
        run_output_path=kwargs.get("run_output_path"),
        build_encoder=lambda: ViCLIPEncoder(
            device=device,
            config=config,
            path_config=path_config,
        ),
        encode_selected_frames=_encode_clip_frames,
        model_log_message="Loading CLIP model for Background Consistency...",
    )
