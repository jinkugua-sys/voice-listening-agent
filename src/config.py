"""配置加载：极简 .env 解析，避免引入额外依赖。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env(path: Path) -> None:
    """读取 .env，已存在的环境变量优先（setdefault）。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class ASRConfig:
    model_size: str = "small"        # tiny / base / small / medium / large-v3
    device: str = "auto"             # auto / cpu / cuda
    compute_type: str = "auto"       # auto / float16 / int8 / int8_float16
    language: str | None = "zh"      # None 表示自动检测
    beam_size: int = 5
    initial_prompt: str = ""


@dataclass
class VADConfig:
    frame_ms: int = 30               # 每帧时长，能量检测的粒度
    start_ratio: float = 2.6         # 起始阈值 = 噪声底 * 该系数
    min_speech_ms: int = 300         # 低于此时长视为噪声
    hangover_ms: int = 600           # 尾部静音超过此时长判定一句结束
    max_segment_s: float = 30.0      # 单句上限，超长强制切分
    noise_floor_min: float = 0.004   # 噪声底下限，防止安静环境阈值过低


@dataclass
class LLMConfig:
    """可选的文本后处理：纠错、断句、摘要。留空即关闭。"""
    base_url: str = ""               # 兼容 OpenAI 协议，如 http://localhost:8000/v1
    api_key: str = ""
    model: str = ""
    enabled: bool = False
    timeout_s: float = 30.0


@dataclass
class AppConfig:
    asr: ASRConfig = field(default_factory=ASRConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    sample_rate: int = 16000
    glossary_path: Path = ROOT / "data" / "glossary.json"
    output_dir: Path = ROOT / "output"


def load_config(env_path: Path | None = None) -> AppConfig:
    _load_env(env_path or (ROOT / ".env"))

    asr = ASRConfig(
        model_size=os.getenv("ASR_MODEL", "small"),
        device=os.getenv("ASR_DEVICE", "auto"),
        compute_type=os.getenv("ASR_COMPUTE_TYPE", "auto"),
        language=(os.getenv("ASR_LANGUAGE") or "zh"),
        beam_size=int(os.getenv("ASR_BEAM_SIZE", "5")),
    )

    vad = VADConfig(
        frame_ms=int(os.getenv("VAD_FRAME_MS", "30")),
        start_ratio=float(os.getenv("VAD_START_RATIO", "2.6")),
        min_speech_ms=int(os.getenv("VAD_MIN_SPEECH_MS", "300")),
        hangover_ms=int(os.getenv("VAD_HANGOVER_MS", "600")),
        max_segment_s=float(os.getenv("VAD_MAX_SEGMENT_S", "30")),
    )

    llm = LLMConfig(
        base_url=os.getenv("LLM_BASE_URL", ""),
        api_key=os.getenv("LLM_API_KEY", ""),
        model=os.getenv("LLM_MODEL", ""),
        enabled=bool(os.getenv("LLM_BASE_URL") and os.getenv("LLM_MODEL")),
        timeout_s=float(os.getenv("LLM_TIMEOUT_S", "30")),
    )

    cfg = AppConfig(asr=asr, vad=vad, llm=llm)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg
