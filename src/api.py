"""HTTP / WebSocket 服务化封装。

启动：
    uvicorn src.api:app --host 0.0.0.0 --port 8000
    （或 uvicorn api:app --app-dir src）

接口：
    GET  /health           健康检查与模型状态
    POST /v1/transcribe    上传音频，返回句级结果
    WS   /v1/stream        推送 16-bit PCM（16kHz 单声道），实时返回听抄结果
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402

from config import load_config  # noqa: E402
from pipeline import Segmenter, Transcriber, transcribe_file  # noqa: E402

cfg = load_config()
transcriber = Transcriber(cfg)

app = FastAPI(
    title="Voice Listening Agent",
    version="0.1.0",
    description="VAD 切分 + faster-whisper + 领域术语后处理的话音听抄服务",
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "asr": {
            "model": cfg.asr.model_size,
            "device": cfg.asr.device,
            "language": cfg.asr.language,
            "loaded": transcriber.asr.loaded,
        },
        "vad": {"frame_ms": cfg.vad.frame_ms, "hangover_ms": cfg.vad.hangover_ms},
        "llm_postprocess": cfg.llm.enabled,
        "glossary_terms": len(transcriber.glossary.terms),
    }


@app.post("/v1/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    """上传整段音频，离线转写。"""
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        utterances = await run_in_threadpool(transcribe_file, cfg, tmp_path, transcriber)
    except Exception as exc:  # noqa: BLE001 - 统一转成 503，便于前端展示
        raise HTTPException(status_code=503, detail=f"转写失败：{str(exc)[:200]}") from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    audio_s = utterances[-1].end_s if utterances else 0.0
    asr_s = sum(u.elapsed_s for u in utterances)
    return {
        "utterances": [
            {"index": u.index, "start": round(u.start_s, 2), "end": round(u.end_s, 2), "text": u.text}
            for u in utterances
        ],
        "audio_seconds": round(audio_s, 2),
        "asr_seconds": round(asr_s, 2),
        "rtf": round(asr_s / audio_s, 3) if audio_s else 0.0,
    }


@app.websocket("/v1/stream")
async def stream(ws: WebSocket) -> None:
    """实时听抄。

    协议：
    - 客户端发二进制：16kHz / 16bit / 单声道 PCM，按帧长（默认 30ms = 960 字节）推送；
    - 客户端发文本 "flush"：立即结束当前句；
    - 服务端回：
        {"type":"speech_start"}                    检测到语音起点（可用于前端打断/提示）
        {"type":"final","text":...,"seconds":...}  一句识别完成
        {"type":"error","detail":...}              识别失败
    """
    await ws.accept()
    seg = Segmenter(cfg.vad, cfg.sample_rate)
    frame_bytes = int(cfg.sample_rate * cfg.vad.frame_ms / 1000) * 2

    starts = {"n": 0}
    seg.on_speech_start = lambda: starts.__setitem__("n", starts["n"] + 1)

    buffer = bytearray()

    async def emit(audio: np.ndarray) -> None:
        try:
            text, elapsed = await run_in_threadpool(transcriber.process, audio)
        except Exception as exc:  # noqa: BLE001 - 返回给前端而不是断连
            await ws.send_json({"type": "error", "detail": str(exc)[:200]})
            return
        if text:
            await ws.send_json(
                {
                    "type": "final",
                    "text": text,
                    "seconds": round(len(audio) / cfg.sample_rate, 2),
                    "elapsed": round(elapsed, 2),
                }
            )

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break

            payload = message.get("bytes")
            if payload is not None:
                buffer.extend(payload)
                while len(buffer) >= frame_bytes:
                    chunk = bytes(buffer[:frame_bytes])
                    del buffer[:frame_bytes]
                    audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0

                    before = starts["n"]
                    finished = seg.push(audio)
                    if starts["n"] > before:
                        await ws.send_json({"type": "speech_start"})
                    for item in finished:
                        await emit(item)
            elif message.get("text") == "flush":
                tail = seg.flush()
                if tail is not None:
                    await emit(tail)
    except WebSocketDisconnect:
        return
