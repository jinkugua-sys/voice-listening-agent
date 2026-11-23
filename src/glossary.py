"""领域术语表：让通用 ASR 认识「专业术语 / 型号 / 缩略语」。

两条路径同时用：
1. initial_prompt：给 Whisper 提供偏置，提升同音字命中率；
2. 后置替换：把仍识别错的固定错法批量纠正。
"""

from __future__ import annotations

import json
from pathlib import Path


class Glossary:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.terms: list[str] = []
        self.replacements: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.terms = list(data.get("terms", []))
        self.replacements = dict(data.get("replacements", {}))

    @property
    def initial_prompt(self) -> str:
        if not self.terms:
            return ""
        return "以下为领域术语，转录时请优先使用标准写法：" + "、".join(self.terms)

    def apply(self, text: str) -> str:
        for wrong, right in self.replacements.items():
            text = text.replace(wrong, right)
        return text
