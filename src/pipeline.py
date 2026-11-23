"""流水线：音频帧 -> VAD 切分 -> ASR -> 术语后处理 -> LLM 纠错 -> 输出。

两种运行模式共用同一套切分与识别逻辑：
- file：离线转写，产出 txt + srt，可量化 RTF；
- mic ：实时听抄，带打断（barge-in）。
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agent import TextAgent
from asr import ASREngine
from config import AppConfig
from glossary import Glossary
from vad import Segmenter


# ---------------------------------------------------------------- 音频读取


def load_audio(path: Path, sample_rate: int = 16000) -> np.ndarray:
    """统一解码为 16k 单声道 float32。

    优先 ffmpeg（兼容 mp3/m4a/视频音轨）；没有 ffmpeg 时对 wav 走标准库兜底，
    保证在干净环境下至少能跑通演示。
    """
    if shutil.which("ffmpeg"):
        return _load_with_ffmpeg(path, sample_rate)
    if path.suffix.lower() == ".wav":
        return _load_wav(path, sample_rate)
    raise RuntimeError("未找到 ffmpeg，且输入不是 wav —— 请安装 ffmpeg 或改用 wav 文件")


def _load_with_ffmpeg(path: Path, sample_rate: int) -> np.ndarray:
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 解码失败：{proc.stderr.decode('utf-8', 'ignore')[:200]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _load_wav(path: Path, sample_rate: int) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"不支持的 wav 位宽：{width * 8} bit")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != sample_rate:
        data = _resample(data, rate, sample_rate)
    return data.astype(np.float32)


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    """线性插值重采样，仅作无 ffmpeg 时的兜底，精度要求高请走 ffmpeg。"""
    n_out = int(round(len(audio) * dst / src))
    if n_out <= 0:
        return audio
    x_src = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_dst = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_dst, x_src, audio).astype(np.float32)


# ---------------------------------------------------------------- 打断控制


class BargeIn:
    """轻量打断机制。

    新语音起点到来时 token 自增，旧任务返回后发现自己 token 过期即丢弃结果，
    从而实现「用户一开口，上一句的等待/输出立刻作废」。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token = 0

    @property
    def token(self) -> int:
        with self._lock:
            return self._token

    def interrupt(self) -> None:
        with self._lock:
            self._token += 1

    def is_stale(self, token: int) -> bool:
        return token != self.token


# ---------------------------------------------------------------- 识别单元


@dataclass
class Utterance:
    index: int
    start_s: float
    end_s: float
    text: str
    elapsed_s: float = 0.0


class Transcriber:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.glossary = Glossary(cfg.glossary_path)
        self.asr = ASREngine(cfg.asr)
        self.agent = TextAgent(cfg.llm)

    def process(self, audio: np.ndarray) -> tuple[str, float]:
        result = self.asr.transcribe(audio, initial_prompt=self.glossary.initial_prompt)
        text = self.glossary.apply(result.text)
        text = self.agent.refine(text, self.glossary.terms)
        return text, result.elapsed_s


# ---------------------------------------------------------------- 离线转写


def transcribe_file(cfg: AppConfig, path: Path, transcriber: Transcriber | None = None) -> list[Utterance]:
    """离线转写。

    transcriber 可复用外部实例（服务端常驻），避免每次请求重复加载模型。
    """
    audio = load_audio(path, cfg.sample_rate)
    seg = Segmenter(cfg.vad, cfg.sample_rate)
    tr = transcriber or Transcriber(cfg)

    frame_len = int(cfg.sample_rate * cfg.vad.frame_ms / 1000)
    utterances: list[Utterance] = []
    offset = 0
    for i in range(0, len(audio) - frame_len + 1, frame_len):
        frame = audio[i : i + frame_len]
        for chunk in seg.push(frame):
            start = offset
            end = offset + len(chunk) / cfg.sample_rate
            text, elapsed = tr.process(chunk)
            utterances.append(Utterance(len(utterances) + 1, start, end, text, elapsed))
            offset = end
    tail = seg.flush()
    if tail is not None:
        start = offset
        end = offset + len(tail) / cfg.sample_rate
        text, elapsed = tr.process(tail)
        utterances.append(Utterance(len(utterances) + 1, start, end, text, elapsed))
    return utterances


def write_outputs(utterances: list[Utterance], stem: Path) -> tuple[Path, Path]:
    txt_path = stem.with_suffix(".txt")
    srt_path = stem.with_suffix(".srt")
    txt_path.write_text("\n".join(u.text for u in utterances if u.text), encoding="utf-8")

    def ts(sec: float) -> str:
        ms = int(round(sec * 1000))
        h, ms = divmod(ms, 3600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = []
    for u in utterances:
        if not u.text:
            continue
        blocks.append(f"{u.index}\n{ts(u.start_s)} --> {ts(u.end_s)}\n{u.text}\n")
    srt_path.write_text("\n".join(blocks), encoding="utf-8")
    return txt_path, srt_path


# ---------------------------------------------------------------- 实时听抄


def run_mic(cfg: AppConfig) -> None:
    import sounddevice as sd  # 仅实时模式需要

    frame_len = int(cfg.sample_rate * cfg.vad.frame_ms / 1000)
    seg = Segmenter(cfg.vad, cfg.sample_rate)
    tr = Transcriber(cfg)
    barge = BargeIn()
    seg.on_speech_start = barge.interrupt  # 新语音起点 -> 作废上一句

    frames: queue.Queue[np.ndarray] = queue.Queue()
    pool = ThreadPoolExecutor(max_workers=1)
    collected: list[str] = []

    def handle(audio: np.ndarray, token: int) -> None:
        try:
            text, elapsed = tr.process(audio)
        except Exception as exc:  # noqa: BLE001 - 单句失败不应拖垮整条流
            print(f"[asr] 识别失败：{exc}")
            return
        if barge.is_stale(token):
            return  # 用户已经开口，丢弃过期结果
        if not text:
            return
        collected.append(text)
        print(f">> {text}   ({elapsed:.2f}s / {len(audio)/cfg.sample_rate:.1f}s 音频)")

    def callback(indata, _frames, _time, status) -> None:  # noqa: ANN001
        if status:
            print(f"[mic] {status}")
        frames.put(indata[:, 0].copy())

    print(f"实时听抄中（模型 {cfg.asr.model_size}，帧长 {cfg.vad.frame_ms}ms），Ctrl+C 结束")
    with sd.InputStream(
        samplerate=cfg.sample_rate,
        channels=1,
        dtype="float32",
        blocksize=frame_len,
        callback=callback,
    ):
        try:
            while True:
                frame = frames.get()
                for chunk in seg.push(frame):
                    pool.submit(handle, chunk, barge.token)
        except KeyboardInterrupt:
            pass

    tail = seg.flush()
    if tail is not None:
        pool.submit(handle, tail, barge.token)
    pool.shutdown(wait=True)

    if collected and tr.agent.enabled:
        summary = tr.agent.summarize(collected)
        if summary:
            print("\n===== 纪要 =====\n" + summary)
