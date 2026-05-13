"""主体一致性：段内 p25 + 跨段代表帧一致性。"""

from narrastream_bench.models.dino_encoder import DINOEncoder
from narrastream_bench.utils.consistency_metric import compute_consistency_metric


def compute_subject_consistency(eval_data, device, config=None, path_config=None, **kwargs):
    return compute_consistency_metric(
        metric_name="subject_consistency",
        eval_data=eval_data,
        device=device,
        config=config,
        path_config=path_config,
        run_output_path=kwargs.get("run_output_path"),
        build_encoder=lambda: DINOEncoder(
            device=device,
            config=config,
            path_config=path_config,
        ),
        encode_selected_frames=lambda encoder, frames: [
            encoder.encode_frame(frame).squeeze(0) for frame in frames
        ],
        model_log_message="Loading DINO model...",
    )
