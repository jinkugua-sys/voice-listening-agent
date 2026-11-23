"""命令行入口。

用法：
    python src/cli.py doctor                    # 环境自检
    python src/cli.py file demo.wav             # 离线转写，产出 txt + srt
    python src/cli.py mic                       # 实时听抄（带打断）
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import AppConfig, load_config  # noqa: E402
from pipeline import transcribe_file, run_mic, write_outputs  # noqa: E402


def cmd_doctor(cfg: AppConfig) -> int:
    ok = True

    print(f"python      : {sys.version.split()[0]}")
    print(f"ffmpeg      : {shutil.which('ffmpeg') or '未安装（file 模式必需）'}")
    ok &= bool(shutil.which("ffmpeg"))

    for mod in ("numpy", "faster_whisper"):
        present = importlib.util.find_spec(mod) is not None
        print(f"{mod:<12}: {'OK' if present else '缺失，pip install -r requirements.txt'}")
        ok &= present

    has_sd = importlib.util.find_spec("sounddevice") is not None
    print(f"{'sounddevice':<12}: {'OK' if has_sd else '缺失（仅 mic 模式需要）'}")

    try:
        import torch  # noqa: F401

        print(f"torch       : {torch.__version__}, cuda={torch.cuda.is_available()}")
    except ImportError:
        print("torch       : 未安装（faster-whisper 不需要 torch，可忽略）")

    print(f"模型配置    : {cfg.asr.model_size} / device={cfg.asr.device} / lang={cfg.asr.language}")
    print(f"LLM 后处理  : {'开启 ' + cfg.llm.model if cfg.llm.enabled else '关闭'}")
    print(f"术语表      : {cfg.glossary_path} ({'存在' if cfg.glossary_path.exists() else '缺失'})")
    return 0 if ok else 1


def cmd_file(cfg: AppConfig, path: Path) -> int:
    if not path.exists():
        print(f"文件不存在：{path}")
        return 2
    started = time.perf_counter()
    utterances = transcribe_file(cfg, path)
    total_audio = utterances[-1].end_s if utterances else 0.0
    txt, srt = write_outputs(utterances, cfg.output_dir / path.stem)

    asr_time = sum(u.elapsed_s for u in utterances)
    for u in utterances:
        print(f"[{u.start_s:6.2f}-{u.end_s:6.2f}] {u.text}")

    print(f"\n句子数 {len(utterances)}｜音频 {total_audio:.1f}s｜识别 {asr_time:.1f}s"
          f"｜RTF {asr_time / total_audio if total_audio else 0:.3f}"
          f"｜总耗时 {time.perf_counter() - started:.1f}s")
    print(f"输出：{txt}\n      {srt}")
    return 0


def cmd_mic(cfg: AppConfig) -> int:
    if importlib.util.find_spec("sounddevice") is None:
        print("缺少 sounddevice，请先 pip install sounddevice")
        return 2
    run_mic(cfg)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="话音听抄 Agent（VAD + ASR + 术语后处理）")
    parser.add_argument("--model", help="覆盖 ASR 模型，如 small / medium / large-v3")
    parser.add_argument("--device", help="覆盖设备，cpu / cuda / auto")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="环境自检")
    p_file = sub.add_parser("file", help="离线转写")
    p_file.add_argument("audio", type=Path)
    sub.add_parser("mic", help="实时听抄")

    args = parser.parse_args()
    cfg = load_config()
    if args.model:
        cfg.asr.model_size = args.model
    if args.device:
        cfg.asr.device = args.device

    if args.cmd == "doctor":
        return cmd_doctor(cfg)
    if args.cmd == "file":
        return cmd_file(cfg, args.audio)
    return cmd_mic(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
