"""默认值：按机器规格自动给线程数，允许配置覆盖。"""

from __future__ import annotations

import os

CAP_OCR_THREADS = 10
PARALLEL_OCR_THREADS = 4     # 与 VLM 并行时的上限（避免抢 CPU）      # 实测 10 线程 1.12s/帧，再多反而变慢（16 线程 1.46s/帧）
CAP_VLM_THREADS = 8


def cpu_count() -> int:
    return os.cpu_count() or 8


def default_ocr_threads() -> int:
    """默认取一半核心，但封顶 10（超过收益为负）。"""
    return max(1, min(CAP_OCR_THREADS, cpu_count() // 2))


def default_vlm_threads() -> int:
    return max(2, min(CAP_VLM_THREADS, cpu_count() // 3))


def default_asr_threads() -> int:
    return max(2, min(8, cpu_count() // 4))


def resolve_threads(configured: int | None, *, kind: str) -> int:
    if configured is not None and int(configured) > 0:
        return int(configured)
    return {
        "ocr": default_ocr_threads,
        "vlm": default_vlm_threads,
        "asr": default_asr_threads,
    }[kind]()


def frame_budget(duration: float) -> int:
    """按时长给视觉帧预算：约每 7 秒 1 帧，夹在 6~15 之间。"""
    if duration <= 0:
        return 6
    return max(6, min(15, round(duration / 7)))


def ocr_budget(duration: float, interval: float = 5.0, cap: int = 90) -> int:
    """OCR 预算：talk 类按间隔密采，但不超过 cap 帧。"""
    if duration <= 0:
        return 2
    return min(cap, max(2, int(duration / interval)))
