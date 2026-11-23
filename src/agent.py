"""可选的文本后处理 Agent。

用 OpenAI 兼容协议（vLLM / Ollama / DeepSeek / 任意云厂商都能接），
不引入 openai SDK，保持零额外依赖。

作用只有两个，且都要求「不改写原意」：
- refine：同音字纠正、术语规范、补标点、去口头语；
- summarize：把多句听抄结果压成纪要。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from config import LLMConfig

_SYSTEM = (
    "你是话音听抄后处理助手。只做三件事：纠正同音字、按给定术语表统一写法、补全标点并去掉口头语。"
    "不得改写内容、不得增删信息、不得解释或总结。直接输出处理后的文本。"
)


class TextAgent:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def _chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "stream": False,
        }
        req = urllib.request.Request(
            self.cfg.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.api_key or 'none'}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data["choices"][0]["message"]["content"] or "").strip()

    def refine(self, text: str, terms: list[str] | None = None) -> str:
        if not self.enabled or not text:
            return text
        hint = f"\n领域术语表：{'、'.join(terms)}" if terms else ""
        try:
            out = self._chat(_SYSTEM, f"待处理听抄文本：{text}{hint}")
            return out or text
        except (urllib.error.URLError, KeyError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"[llm] 后处理跳过（{exc}）")
            return text

    def summarize(self, texts: list[str]) -> str:
        if not self.enabled or not texts:
            return ""
        joined = "\n".join(texts)
        try:
            return self._chat(
                "你是会议纪要助手，用不超过 5 条要点总结，保留数字与型号。",
                joined,
            )
        except (urllib.error.URLError, KeyError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"[llm] 摘要跳过（{exc}）")
            return ""
