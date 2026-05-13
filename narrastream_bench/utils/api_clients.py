"""OpenAI-compatible API client helpers."""
import os
from dataclasses import dataclass

from openai import OpenAI


DEFAULT_MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1/"
DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"


@dataclass(frozen=True)
class ServiceConfig:
    provider: str
    model: str
    base_url: str | None
    api_key: str | None
    api_key_env: str | None
    max_tokens: int
    temperature: float


DEFAULT_SERVICE_CONFIGS = {
    "mllm": {
        "provider": "siliconflow",
        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "base_url": DEFAULT_SILICONFLOW_BASE_URL,
        "api_key_env": "SILICONFLOW_API_KEY",
        "max_tokens": 256,
        "temperature": 0.0,
    },
    "vlm": {
        "provider": "siliconflow",
        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "base_url": DEFAULT_SILICONFLOW_BASE_URL,
        "api_key_env": "SILICONFLOW_API_KEY",
        "max_tokens": 32,
        "temperature": 0.0,
    },
    "planner": {
        "provider": "siliconflow",
        "model": "Qwen/Qwen3.5-27B",
        "base_url": DEFAULT_SILICONFLOW_BASE_URL,
        "api_key_env": "SILICONFLOW_API_KEY",
        "max_tokens": 256,
        "temperature": 0.0,
    },
}


def resolve_service_config(config=None, service_name="mllm", model_override=None):
    resolved = dict(DEFAULT_SERVICE_CONFIGS.get(service_name, {}))
    service_cfg = ((config or {}).get("services") or {}).get(service_name, {})
    resolved.update(service_cfg)
    if model_override:
        resolved["model"] = model_override

    provider = resolved.get("provider", "openai")
    base_url = resolved.get("base_url")
    if provider == "modelscope" and not base_url:
        base_url = DEFAULT_MODELSCOPE_BASE_URL
    if provider == "siliconflow" and not base_url:
        base_url = DEFAULT_SILICONFLOW_BASE_URL

    api_key_env = resolved.get("api_key_env")
    api_key = resolved.get("api_key")
    if not api_key and api_key_env:
        api_key = os.getenv(api_key_env)
    if not api_key and provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")

    return ServiceConfig(
        provider=provider,
        model=resolved["model"],
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        max_tokens=int(resolved.get("max_tokens", 256)),
        temperature=float(resolved.get("temperature", 0.0)),
    )


def build_openai_compatible_client(service_config: ServiceConfig):
    kwargs = {}
    if service_config.api_key:
        kwargs["api_key"] = service_config.api_key
    if service_config.base_url:
        kwargs["base_url"] = service_config.base_url
    return OpenAI(**kwargs)
