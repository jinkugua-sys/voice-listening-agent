"""ASR 引擎封装：faster-whisper + 设备/精度自适应。

延迟与精度可调：
- model_size 越大越准、越慢，演示用 small/medium 即可；
- GPU 上 float16，CPU 上 int8，可省一半内存。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from config import ASRConfig


@dataclass
class ASRResult:
    text: str
    elapsed_s: float
    audio_s: float

    @property
    def rtf(self) -> float:
        """实时率，越小越快（<1 表示比实时快）。"""
        return self.elapsed_s / self.audio_s if self.audio_s else 0.0


class ASREngine:
    def __init__(self, cfg: ASRConfig) -> None:
        self.cfg = cfg
        self._model = None
        self.device_info = ""

    @property
    def loaded(self) -> bool:
        """模型是否已加载（用于健康检查，不触发加载）。"""
        return self._model is not None

    def _candidates(self) -> list[tuple[str, str]]:
        """按优先级返回 (device, compute_type) 组合。"""
        device = self.cfg.device.lower()
        compute = self.cfg.compute_type.lower()
        if device == "auto":
            pairs = [("cuda", "float16"), ("cpu", "int8")]
        elif device == "cuda":
            pairs = [("cuda", compute if compute != "auto" else "float16")]
        else:
            pairs = [("cpu", compute if compute != "auto" else "int8")]
        return pairs

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # 延迟导入，避免无依赖时报错

        last_err: Exception | None = None
        for device, compute in self._candidates():
            try:
                self._model = WhisperModel(self.cfg.model_size, device=device, compute_type=compute)
                self.device_info = f"{device}/{compute}/{self.cfg.model_size}"
                return
            except Exception as exc:  # noqa: BLE001 - 需要回退到下一组
                last_err = exc
        raise RuntimeError(f"模型加载失败，请检查 CUDA / 依赖：{last_err}") from last_err

    def transcribe(self, audio: np.ndarray, initial_prompt: str = "") -> ASRResult:
        self._ensure_model()
        assert self._model is not None
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        audio_s = len(audio) / 16000

        started = time.perf_counter()
        segments, _info = self._model.transcribe(
            audio,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            vad_filter=False,            # 已在切分阶段做过 VAD，避免二次截断
            condition_on_previous_text=False,  # 逐句独立，减少串话与重复
            initial_prompt=initial_prompt or None,
        )
        text = "".join(seg.text for seg in segments).strip()
        return ASRResult(text=text, elapsed_s=time.perf_counter() - started, audio_s=audio_s)
