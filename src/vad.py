"""VAD 与语句切分。

设计取舍：
- 不依赖 webrtcvad / silero，用自适应能量阈值即可满足演示，且零额外依赖；
- 噪声底用慢速 EMA 跟踪，避免固定阈值在不同麦克风/环境下失效；
- 切分器对外只暴露「帧进、句子出」，便于后续替换成 Silero VAD 而不改上层。
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from config import VADConfig


class EnergyVAD:
    """自适应能量 VAD。"""

    def __init__(self, cfg: VADConfig) -> None:
        self.cfg = cfg
        self.noise_floor = cfg.noise_floor_min

    def is_speech(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float32)) + 1e-12))
        threshold = max(self.cfg.noise_floor_min, self.noise_floor * self.cfg.start_ratio)
        speech = rms > threshold
        if not speech:
            # 只在静音段更新噪声底，避免把语音算进噪声
            self.noise_floor = 0.98 * self.noise_floor + 0.02 * rms
        return speech


class Segmenter:
    """把连续音频帧切成「一句话」。

    push() 返回本帧触发的已完成语句（正常每帧返回空列表）。
    on_speech_start 在检测到新语音起点时回调，用于打断上一句的处理。
    """

    def __init__(self, cfg: VADConfig, sample_rate: int) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.vad = EnergyVAD(cfg)
        self.on_speech_start: Callable[[], None] | None = None
        self._buffers: list[np.ndarray] = []
        self._speech_ms = 0
        self._silence_ms = 0
        self._in_speech = False

    def _reset(self) -> None:
        self._buffers = []
        self._speech_ms = 0
        self._silence_ms = 0
        self._in_speech = False

    def push(self, frame: np.ndarray) -> list[np.ndarray]:
        finished: list[np.ndarray] = []
        speech = self.vad.is_speech(frame)

        if not self._in_speech:
            if not speech:
                return finished
            # 语音起点
            self._in_speech = True
            self._buffers = [frame.copy()]
            self._speech_ms = self.cfg.frame_ms
            self._silence_ms = 0
            if self.on_speech_start:
                self.on_speech_start()
            return finished

        self._buffers.append(frame.copy())
        if speech:
            self._speech_ms += self.cfg.frame_ms
            self._silence_ms = 0
        else:
            self._silence_ms += self.cfg.frame_ms

        total_ms = len(self._buffers) * self.cfg.frame_ms
        ended = self._silence_ms >= self.cfg.hangover_ms
        too_long = total_ms >= self.cfg.max_segment_s * 1000

        if ended or too_long:
            if self._speech_ms >= self.cfg.min_speech_ms:
                finished.append(np.concatenate(self._buffers))
            self._reset()
        return finished

    def flush(self) -> np.ndarray | None:
        """结束录音时把残留语音吐出来。"""
        if self._in_speech and self._speech_ms >= self.cfg.min_speech_ms:
            audio = np.concatenate(self._buffers) if self._buffers else None
            self._reset()
            return audio
        self._reset()
        return None
