"""用 ffmpeg 场景检测统计「画面大变化」，替代/补充像素差判型。

原理：ffmpeg 的 scene 分数是相邻帧的直方图差异（0~1）。只统计超过阈值的帧，
得到「切点时间 + 分数」，既用于判类型，也直接当抽帧候选点（§3.2：只用来找变化点）。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

TIME_RE = re.compile(r"pts_time:([0-9.]+)")
SCORE_RE = re.compile(r"lavfi\.scene_score=([0-9.]+)")


def scene_cuts(
    video: Path,
    *,
    ffmpeg: str = "",
    threshold: float = 0.3,
    max_cuts: int = 200,
) -> list[dict]:
    """返回 [{t, score}]，按时间升序。"""
    from .media import _bin, DEFAULT_FFMPEG

    exe = _bin(ffmpeg or "", DEFAULT_FFMPEG)
    # 直接传 argv 时不能带 shell 引号；逗号在 select 表达式里要转义
    # 先缩到 320 宽再算场景分：分数几乎不变，解码快很多
    filter_arg = f"scale=320:-2,select=gt(scene\\,{threshold}),metadata=print"
    cmd = [
        exe, "-hide_banner", "-loglevel", "info",
        "-i", str(video), "-an",
        "-filter:v", filter_arg,
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=600)
    text = (proc.stderr or b"").decode("utf-8", "replace")

    cuts: list[dict] = []
    pending_t: float | None = None
    for line in text.splitlines():
        t_match = TIME_RE.search(line)
        if t_match:
            pending_t = float(t_match.group(1))
            continue
        s_match = SCORE_RE.search(line)
        if s_match and pending_t is not None:
            cuts.append({"t": round(pending_t, 3), "score": round(float(s_match.group(1)), 3)})
            pending_t = None
            if len(cuts) >= max_cuts:
                break
    return cuts


def scene_stats(cuts: list[dict], duration: float) -> dict:
    """把切点换算成可判定的密度指标。"""
    if duration <= 0:
        return {"cuts": len(cuts), "per_minute": 0.0, "mean_score": 0.0}
    minutes = max(duration / 60.0, 1e-6)
    scores = [c["score"] for c in cuts]
    return {
        "cuts": len(cuts),
        "per_minute": round(len(cuts) / minutes, 2),
        "mean_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
    }
