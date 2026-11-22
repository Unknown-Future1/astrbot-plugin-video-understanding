"""抽帧策略：粗筛判类型 → 按段加量 → 按语音时间戳精抽。

思路（对应 §3.2 与待决策 Q14）：
  ① 均匀粗抽 N 帧，只算感知哈希，不调模型
  ② 画面变化小 → talk（座谈/口播/单场景）：视觉少抽，OCR 与语音加密
     画面变化大 → dynamic：按时间窗加量，再由语音时间戳定点精抽
"""

from __future__ import annotations

import math
import re
from pathlib import Path

TALK_THRESHOLD = 0.12          # 相邻帧像素平均差，低于它视为静态/口播类
REFINE_KEYWORDS = ("元", "万", "年", "月", "日", "秒", "度", "%", "折扣", "价格")
_QUOTE = re.compile(r"[「」“”\"《》]")
_DIGIT = re.compile(r"\d")


def dhash(image_path: Path, size: int = 8) -> int:
    """8x8 差分哈希；只依赖 PIL，不引入额外库。"""
    from PIL import Image

    with Image.open(image_path) as img:
        gray = img.convert("L").resize((size + 1, size), Image.LANCZOS)
        pixels = list(gray.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            left = pixels[row * (size + 1) + col]
            right = pixels[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def distance(a: int, b: int) -> float:
    return bin(a ^ b).count("1") / 64.0


def pixel_diff(a: Path, b: Path) -> float:
    """缩到 64x36 灰度后求平均绝对差，抗轻微位移，比哈希稳。"""
    from PIL import Image, ImageChops, ImageStat

    with Image.open(a) as ia, Image.open(b) as ib:
        ga = ia.convert("L").resize((64, 36), Image.LANCZOS)
        gb = ib.convert("L").resize((64, 36), Image.LANCZOS)
        diff = ImageChops.difference(ga, gb)
        return ImageStat.Stat(diff).mean[0] / 255.0


def classify(files: list[Path]) -> dict:
    """按相邻帧像素差判断类型：talk（静态/口播/幻灯片）或 dynamic。"""
    diffs = [pixel_diff(a, b) for a, b in zip(files, files[1:])]
    if not diffs:
        return {"profile": "talk", "mean": 0.0, "max": 0.0, "samples": len(files), "diffs": []}
    mean = sum(diffs) / len(diffs)
    profile = "talk" if mean <= TALK_THRESHOLD else "dynamic"
    return {
        "profile": profile,
        "mean": round(mean, 3),
        "max": round(max(diffs), 3),
        "samples": len(files),
        "diffs": [round(d, 3) for d in diffs],
    }


def static_keep(times: list[float], diffs: list[float], threshold: float = 0.05) -> list[float]:
    """talk 类只保留"画面确实变了"的点（外加第一帧）。"""
    keep = [times[0]] if times else []
    for index, diff in enumerate(diffs, start=1):
        if diff >= threshold and index < len(times):
            keep.append(times[index])
    return keep


def coarse_times(duration: float, count: int = 8) -> list[float]:
    if duration <= 0:
        return []
    count = max(3, count)
    span = duration * 0.9
    start = duration * 0.05
    step = span / count
    return [round(start + step * (i + 0.5), 3) for i in range(count)]


def ocr_times_talk(duration: float, interval: float = 5.0, cap: int = 90) -> list[float]:
    """talk 类：OCR 便宜，按固定间隔密采（默认 5 秒一张）。"""
    if duration <= 0:
        return []
    count = min(cap, max(2, int(math.ceil(duration / interval))))
    return coarse_times(duration, count)


def window_times(duration: float, window: float = 30.0, per_window: int = 2) -> list[float]:
    """dynamic 类：每个时间窗内再取 per_window 个点。"""
    times: list[float] = []
    start = 0.0
    while start < duration:
        end = min(start + window, duration)
        span = end - start
        for index in range(per_window):
            times.append(round(start + span * (index + 0.5) / per_window, 3))
        start = end
    return times


def refine_times(
    speech_entries: list[dict],
    taken: list[float],
    *,
    cap: int = 8,
    min_gap: float = 2.5,
) -> list[dict]:
    """按语音时间戳定点精抽：含数字/引号/单位/价格的句子优先。"""
    picked: list[dict] = []
    used = list(taken)
    for entry in speech_entries:
        if len(picked) >= cap:
            break
        text = entry.get("text") or ""
        score = 0
        if _DIGIT.search(text):
            score += 2
        if _QUOTE.search(text):
            score += 2
        if any(word in text for word in REFINE_KEYWORDS):
            score += 1
        if score < 2:
            continue
        t = float(entry.get("t0") or 0.0)
        if any(abs(t - other) < min_gap for other in used):
            continue
        used.append(t)
        picked.append(
            {
                "t": round(t + 0.05, 3),
                "reason": "has_digits" if _DIGIT.search(text) else "has_quote",
                "speech_id": entry.get("id"),
                "ask": f"画面是否印证：{text[:24]}",
            }
        )
    return picked


def merge_unique(times: list[float], extra: list[float], min_gap: float = 1.5) -> list[float]:
    merged = sorted(times)
    for t in sorted(extra):
        if all(abs(t - other) >= min_gap for other in merged):
            merged.append(t)
    return sorted(merged)
