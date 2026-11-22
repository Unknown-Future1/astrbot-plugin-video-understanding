"""术语纠错：语音转写用画面文字里的同音词纠正（稳真 → 稳帧）。

只处理「2~4 字、拼音相同、字不同、且至少一个字不同」的片段，
避免大改；纠正结果记进 bundle，方便核对。
"""

from __future__ import annotations

import re

try:
    from pypinyin import lazy_pinyin
except Exception:  # noqa: BLE001
    lazy_pinyin = None

CJK_ONLY = re.compile(r"^[\u4e00-\u9fff]+$")


def _pinyin(text: str) -> tuple[str, ...] | None:
    if lazy_pinyin is None or not text:
        return None
    try:
        return tuple(lazy_pinyin(text))
    except Exception:  # noqa: BLE001
        return None


def _candidates(lines: list[str]) -> dict[tuple[str, ...], str]:
    """画面文字里的 2~4 字词，按拼音建索引。"""
    table: dict[tuple[str, ...], str] = {}
    for line in lines:
        text = line.strip()
        for size in (4, 3, 2):
            for i in range(len(text) - size + 1):
                word = text[i : i + size]
                if not CJK_ONLY.match(word):
                    continue
                py = _pinyin(word)
                if py:
                    table.setdefault(py, word)
    return table


def correct_speech(speech_texts: list[str], screen_lines: list[str]) -> tuple[list[str], list[dict]]:
    """返回（纠正后的语音, 纠正记录）。"""
    table = _candidates(screen_lines)
    if not table:
        return speech_texts, []
    fixed: list[str] = []
    corrections: list[dict] = []
    for text in speech_texts:
        out = text
        for size in (4, 3, 2):
            i = 0
            while i <= len(out) - size:
                word = out[i : i + size]
                if CJK_ONLY.match(word):
                    py = _pinyin(word)
                    target = table.get(py) if py else None
                    if target and target != word:
                        out = out[:i] + target + out[i + size :]
                        corrections.append({"from": word, "to": target})
                        i += size
                        continue
                i += 1
        fixed.append(out)
    # 去重
    seen = set()
    unique = []
    for item in corrections:
        key = (item["from"], item["to"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return fixed, unique
