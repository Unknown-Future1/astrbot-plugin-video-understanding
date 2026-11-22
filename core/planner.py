"""把 ffmpeg 场景切点接入判型与抽帧计划。

判型规则（实测校准）：
  · 切点密度 ≥ 8 次/分钟  或  像素均值 ≥ 0.12  → dynamic
  · 其余 → talk（static / 口播 / 单场景 + 字幕）
切点本身直接作为 dynamic 的候选抽帧点（按分数取，注意时间分散）。
"""

from __future__ import annotations

from pathlib import Path

from . import scene, sampling


def classify(video: Path, duration: float, coarse_files: list[Path], cuts: list[dict]) -> dict:
    stats = scene.scene_stats(cuts, duration)
    diffs = [sampling.pixel_diff(a, b) for a, b in zip(coarse_files, coarse_files[1:])]
    pixel_mean = sum(diffs) / len(diffs) if diffs else 0.0

    dynamic = stats["per_minute"] >= 8.0 or pixel_mean >= 0.12
    return {
        "profile": "dynamic" if dynamic else "talk",
        "cuts": stats["cuts"],
        "cuts_per_minute": stats["per_minute"],
        "mean_score": stats["mean_score"],
        "pixel_mean": round(pixel_mean, 3),
        "pixel_max": round(max(diffs), 3) if diffs else 0.0,
        "samples": len(coarse_files),
        "diffs": [round(d, 3) for d in diffs],
    }


def cut_candidates(cuts: list[dict], duration: float, cap: int = 10) -> list[float]:
    """按分数挑切点，同时保证时间上铺开。"""
    if not cuts or cap <= 0:
        return []
    ranked = sorted(cuts, key=lambda item: item["score"], reverse=True)
    picked: list[float] = []
    min_gap = max(1.5, duration / (cap * 2))
    for item in ranked:
        t = item["t"]
        if all(abs(t - other) >= min_gap for other in picked):
            picked.append(round(t, 3))
        if len(picked) >= cap:
            break
    return sorted(picked)


def plan_dynamic(
    duration: float,
    cuts: list[dict],
    coarse: list[float] | None = None,
    *,
    cap: int = 12,
) -> list[float]:
    """dynamic：粗筛点 + 切点 + 窗口补齐，按时间排序并限上限。"""
    times = list(coarse or [])
    times += cut_candidates(cuts, duration, cap=cap // 2)
    times += sampling.window_times(duration, 30.0, 1)
    merged = sampling.merge_unique(times, [], min_gap=1.0)
    if len(merged) > cap:
        step = len(merged) / cap
        merged = [merged[int(i * step)] for i in range(cap)]
    return sorted(merged)


def plan_timeline(
    duration: float,
    cuts: list[dict],
    *,
    cap: int = 9,
    min_gap: float = 2.5,
) -> list[float]:
    """把时长切成 cap 段，每段取「分数最高的切点」，没有就取段中点。

    相比"全局取 top 切点"，这样不会把 3 帧都落在同一次跳转上，
    也不会漏掉中间没有明显切点但内容重要的段落。
    """
    if duration <= 0 or cap <= 0:
        return []
    cap = max(3, cap)
    seg = duration / cap
    picked: list[float] = []

    # 固定采样：开头 1 秒内必有帧（很多视频的关键信息在开头），结尾前 1 秒也看一眼
    opening = round(min(0.8, duration * 0.02), 3)
    ending = round(max(opening, duration - min(1.0, duration * 0.03)), 3)
    for index in range(cap):
        start, end = index * seg, min((index + 1) * seg, duration)
        inside = [c for c in cuts if start <= c["t"] < end]
        if inside:
            best = max(inside, key=lambda c: c["score"])
            t = best["t"]
        else:
            t = start + (end - start) / 2
        if picked and abs(t - picked[-1]) < min_gap:
            t = min(end - 0.1, t + min_gap)
        if t < duration:
            picked.append(round(t, 3))
    # 去重（保护）
    deduped: list[float] = []
    for t in picked:
        if not deduped or abs(t - deduped[-1]) >= min_gap:
            deduped.append(t)

    for anchor in (opening, ending):
        if all(abs(anchor - t) > 1.5 for t in deduped) and anchor < duration:
            deduped.append(anchor)
    return sorted(deduped)[: cap + 2]
