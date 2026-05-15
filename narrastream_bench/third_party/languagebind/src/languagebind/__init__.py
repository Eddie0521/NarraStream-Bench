from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig

try:
    from .image.configuration_image import LanguageBindImageConfig
    from .image.modeling_image import LanguageBindImage
    from .image.tokenization_image import LanguageBindImageTokenizer
    from .image.processing_image import LanguageBindImageProcessor
except Exception:
    LanguageBindImageConfig = None
    LanguageBindImage = None
    LanguageBindImageTokenizer = None
    LanguageBindImageProcessor = None

from .video.configuration_video import LanguageBindVideoConfig
from .video.modeling_video import LanguageBindVideo
from .video.tokenization_video import LanguageBindVideoTokenizer
from .video.processing_video import LanguageBindVideoProcessor

try:
    from .depth.configuration_depth import LanguageBindDepthConfig
    from .depth.modeling_depth import LanguageBindDepth
    from .depth.tokenization_depth import LanguageBindDepthTokenizer
    from .depth.processing_depth import LanguageBindDepthProcessor
except Exception:
    LanguageBindDepthConfig = None
    LanguageBindDepth = None
    LanguageBindDepthTokenizer = None
    LanguageBindDepthProcessor = None

try:
    from .audio.configuration_audio import LanguageBindAudioConfig
    from .audio.modeling_audio import LanguageBindAudio
    from .audio.tokenization_audio import LanguageBindAudioTokenizer
    from .audio.processing_audio import LanguageBindAudioProcessor
except Exception:
    LanguageBindAudioConfig = None
    LanguageBindAudio = None
    LanguageBindAudioTokenizer = None
    LanguageBindAudioProcessor = None

try:
    from .thermal.configuration_thermal import LanguageBindThermalConfig
    from .thermal.modeling_thermal import LanguageBindThermal
    from .thermal.tokenization_thermal import LanguageBindThermalTokenizer
    from .thermal.processing_thermal import LanguageBindThermalProcessor
except Exception:
    LanguageBindThermalConfig = None
    LanguageBindThermal = None
    LanguageBindThermalTokenizer = None
    LanguageBindThermalProcessor = None



config_dict = {
    'video': LanguageBindVideoConfig,
}
model_dict = {
    'video': LanguageBindVideo,
}
transform_dict = {
    'video': LanguageBindVideoProcessor,
}

if LanguageBindImageConfig is not None:
    config_dict['image'] = LanguageBindImageConfig
    model_dict['image'] = LanguageBindImage
    transform_dict['image'] = LanguageBindImageProcessor

if LanguageBindDepthConfig is not None:
    config_dict['depth'] = LanguageBindDepthConfig
    model_dict['depth'] = LanguageBindDepth
    transform_dict['depth'] = LanguageBindDepthProcessor

if LanguageBindAudioConfig is not None:
    config_dict['audio'] = LanguageBindAudioConfig
    model_dict['audio'] = LanguageBindAudio
    transform_dict['audio'] = LanguageBindAudioProcessor

if LanguageBindThermalConfig is not None:
    config_dict['thermal'] = LanguageBindThermalConfig
    model_dict['thermal'] = LanguageBindThermal
    transform_dict['thermal'] = LanguageBindThermalProcessor


def _force_eager_attn(module):
    if hasattr(module, "config") and getattr(module, "config", None) is not None:
        try:
            module.config._attn_implementation = "eager"
        except Exception:
            pass
    for child in module.children():
        _force_eager_attn(child)

class LanguageBind(nn.Module):
    def __init__(self, clip_type, use_temp=True, cache_dir='./cache_dir'):
        super(LanguageBind, self).__init__()
        self.use_temp = use_temp
        self.modality_encoder = {}
        self.modality_proj = {}
        self.modality_scale = {}
        self.modality_config = {}
        for k, v in clip_type.items():
            load_kwargs = {"low_cpu_mem_usage": True}
            resolved_path = Path(v).expanduser()
            if resolved_path.exists():
                pretrained_ckpt = str(resolved_path)
                load_kwargs["local_files_only"] = True
            elif "/" in v:
                pretrained_ckpt = v
            else:
                pretrained_ckpt = f'LanguageBind/{v}'
            model = model_dict[k].from_pretrained(
                pretrained_ckpt,
                cache_dir=cache_dir,
                **load_kwargs,
            )
            _force_eager_attn(model)
            self.modality_encoder[k] = model.vision_model
            self.modality_proj[k] = model.visual_projection
            self.modality_scale[k] = model.logit_scale
            self.modality_config[k] = model.config
        self.modality_encoder['language'] = model.text_model
        self.modality_proj['language'] = model.text_projection

        self.modality_encoder = nn.ModuleDict(self.modality_encoder)
        self.modality_proj = nn.ModuleDict(self.modality_proj)

    def forward(self, inputs):
        outputs = {}
        for key, value in inputs.items():
            value = self.modality_encoder[key](**value)[1]
            value = self.modality_proj[key](value)
            value = value / value.norm(p=2, dim=-1, keepdim=True)
            if self.use_temp:
                if key != 'language':
                    value = value * self.modality_scale[key].exp()
            outputs[key] = value
        return outputs

def to_device(x, device):
    out_dict = {k: v.to(device) for k, v in x.items()}
    return out_dict
