# 话音听抄 Agent（Voice Listening Agent）

面向**专业术语场景**的实时语音转写流水线：VAD 切分 → faster-whisper 识别 → 领域术语后处理 → 可选 LLM 纠错，支持实时听抄与打断。

本项目来自真实工程需求：通用 ASR 在「专业术语 / 设备型号 / 缩略语」上错得离谱，而把领域术语表同时注入到 `initial_prompt` 与后置替换，能在不重训模型的前提下明显降低同音字错误。

---

## 一、它能做什么

| 能力 | 对应岗位要求 |
|---|---|
| 实时麦克风听抄 + 一句一输出 | 现场技术支持、语音产品交付 |
| 离线批量转写，产出 txt / srt | 测试记录整理、验收材料生成 |
| 领域术语表（热词注入 + 纠错替换） | 通信 / 电子 / 航空等垂直场景落地 |
| 打断（barge-in）：用户开口即作废上一句 | 对话式 AI / 语音 Agent 交付 |
| LLM 后处理可开关，断网可降级 | 私有化部署、边缘环境交付 |
| RTF 实时率统计 | 性能指标量化，便于向客户证明 |

## 二、快速开始

```bash
# 1) 装依赖
pip install -r requirements.txt

# 2) 自检（检查 ffmpeg / 依赖 / GPU）
python src/cli.py doctor

# 3) 离线转写（任意 ffmpeg 能解的格式：wav/mp3/m4a/视频音轨）
python src/cli.py file path/to/audio.wav

# 4) 实时听抄（Ctrl+C 结束，结束后若开启 LLM 会输出纪要）
python src/cli.py mic

# 5) 起 HTTP 服务（可选，需再装服务化依赖）
pip install -r requirements-api.txt
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

输出在 `output/` 下：同名 `.txt` 与 `.srt`。控制台同时打印句级时间戳、单句耗时与整段 RTF。

可选：复制 `.env.example` 为 `.env`，接本地 vLLM / Ollama 做后处理

```ini
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:7b
```

不配也能跑，LLM 相关步骤自动跳过。

## 三、目录结构

```
voice-listening-agent/
├── src/
│   ├── config.py      # .env 解析与配置数据类
│   ├── vad.py         # 自适应能量 VAD + 语句切分器
│   ├── asr.py         # faster-whisper 封装，设备/精度自适应
│   ├── glossary.py    # 领域术语表：热词注入 + 替换
│   ├── agent.py       # 可选 LLM 后处理（OpenAI 兼容协议）
│   ├── pipeline.py    # 流水线编排 + 打断控制 + srt 输出
│   ├── api.py         # FastAPI：HTTP 上传转写 + WebSocket 实时流
│   └── cli.py         # doctor / file / mic 三个入口
├── data/glossary.json # 领域术语表（可替换为你自己的领域）
├── docs/architecture.md
├── Dockerfile         # 含 ffmpeg，模型缓存可挂载
└── requirements*.txt
```

## 三点五、服务化接口

| 接口 | 说明 |
|---|---|
| `GET /health` | 模型状态、VAD 参数、术语表规模，可直接做健康检查 |
| `POST /v1/transcribe` | 上传音频（multipart），返回句级结果与 RTF |
| `WS /v1/stream` | 推送 16k/16bit/单声道 PCM，实时返回 `speech_start` 与 `final` |

```bash
# 上传转写
curl -X POST http://127.0.0.1:8000/v1/transcribe -F "file=@demo.wav"

# 返回
# {"utterances":[{"index":1,"start":0,"end":3.0,"text":"..."}],
#  "audio_seconds":3.0,"asr_seconds":0.42,"rtf":0.14}
```

WebSocket 协议（前端接入用）：

```text
客户端 -> 二进制帧：每帧 30ms，即 960 字节 16bit PCM
客户端 -> 文本 "flush"：立即结束当前句
服务端 -> {"type":"speech_start"}                    检测到语音起点，可用于 UI 打断反馈
服务端 -> {"type":"final","text":..,"seconds":..}    一句识别完成
服务端 -> {"type":"error","detail":..}               识别异常（不断连）
```

Docker 部署：

```bash
docker build -t voice-listening-agent .
docker run -p 8000:8000 -v ${PWD}/models:/models voice-listening-agent
```

挂载 `/models` 可复用已下载权重，避免容器重建后重复拉取。

## 四、为什么这样设计

详见 [docs/architecture.md](docs/architecture.md)，三个关键取舍：

1. **先切分再识别**，不用 whisper 自带的 `vad_filter`，避免二次截断丢句首；
2. **自我实现能量 VAD**，零额外依赖、可解释，接口隔离便于换 Silero；
3. **LLM 只做纠错，不做理解**，失败自动降级为原文，保证断网可演示。

## 五、演示脚本（面谈时 3 分钟讲完）

1. 打开 `python src/cli.py mic`，正常说一句 → 1 秒内出字；
2. 在它还在识别时立刻开口 → 上一句结果被丢弃，证明打断生效；
3. 说一句含术语的话（如「相控阵接收天线的信噪比」）→ 对比关掉 `data/glossary.json` 前后的错字差异；
4. 跑 `python src/cli.py file long.wav` → 展示 RTF 数字，说明"10 分钟音频 X 秒出稿"。

## 六、后续路线

- [ ] 换 Silero VAD，提升嘈杂环境鲁棒性
- [ ] 接流式 ASR（如 FunASR Streaming / whisper-streaming），把首字延迟压到 300ms 内
- [ ] 加 TTS 与播放，变成完整语音对话 Agent（可迁移到 pipecat）
- [ ] 用 FastAPI 包装 `Transcriber`，提供 HTTP 接口
- [ ] Docker 化，支持离线镜像交付

## 七、环境要求

- Python 3.10+
- ffmpeg（可选）：装了则支持 mp3/m4a/视频音轨；**没有 ffmpeg 也能跑 wav**（内置标准库解码 + 重采样兜底）
- 可选：NVIDIA GPU + CUDA（自动启用 float16；无 GPU 用 CPU int8 亦可运行）

## 八、已知限制（面试时主动说明，比被问出来好）

1. **能量 VAD 对强噪声不鲁棒**：车载、风噪、多人说话场景会误切。升级路径已备好，换 Silero VAD 即可，不影响上层。
2. **句尾包含 hangover 静音**：切出的语音段比真实说话长 600ms，因此 SRT 时间戳略宽。要精确时间戳可换成 whisper 自带的 segment 时间。
3. **非流式 ASR**：当前是"一句一识别"，首字延迟等于整句时长。要做真正常驻对话，需要换流式 ASR 或迁移到 pipecat。

