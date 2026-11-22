"""媒体处理：探测、抽音轨、抽帧。

所有时间都是「原视频第几秒」（float，3 位小数），与 ARCHITECTURE §4.7 一致。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FFMPEG = r"H:\Python\tools\ffmpeg\bin\ffmpeg.exe"
DEFAULT_FFPROBE = r"H:\Python\tools\ffmpeg\bin\ffprobe.exe"


@dataclass
class VideoInfo:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    rotated: bool

    def as_dict(self) -> dict:
        return {
            "duration": round(self.duration, 3),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "has_audio": self.has_audio,
            "rotated": self.rotated,
        }


def _bin(configured: str, fallback: str) -> str:
    if configured and Path(configured).is_file():
        return configured
    found = shutil.which(Path(fallback).name)
    return found or fallback


def run(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def probe(video: Path, *, ffprobe: str = "") -> VideoInfo:
    exe = _bin(ffprobe or "", DEFAULT_FFPROBE)
    out = run(
        [exe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)]
    ).stdout.decode("utf-8", "replace")
    data = json.loads(out or "{}")

    duration = float((data.get("format") or {}).get("duration") or 0.0)
    width = height = 0
    fps = 0.0
    has_audio = False
    rotated = False
    for stream in data.get("streams") or []:
        codec_type = stream.get("codec_type")
        if codec_type == "video" and not width:
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
            rate = stream.get("avg_frame_rate") or "0/1"
            try:
                num, den = rate.split("/")
                fps = float(num) / float(den) if float(den) else 0.0
            except (ValueError, ZeroDivisionError):
                fps = 0.0
            rotation = (stream.get("tags") or {}).get("rotate")
            if rotation and abs(int(float(rotation))) % 180 == 90:
                rotated = True
                width, height = height, width
        elif codec_type == "audio":
            has_audio = True
    return VideoInfo(duration, width, height, fps, has_audio, rotated)


def extract_audio(video: Path, wav: Path, *, ffmpeg: str = "") -> Path:
    exe = _bin(ffmpeg, DEFAULT_FFMPEG)
    wav.parent.mkdir(parents=True, exist_ok=True)
    run(
        [exe, "-y", "-loglevel", "error", "-i", str(video), "-vn",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)]
    )
    return wav


def sample_times(duration: float, count: int) -> list[float]:
    """均匀取点，避开首尾各 5%，保证不会停在片头黑场或片尾。"""
    if duration <= 0 or count <= 0:
        return []
    span = duration * 0.9
    start = duration * 0.05
    step = span / count
    return [round(start + step * (i + 0.5), 3) for i in range(count)]


def windows(duration: float, size: float = 30.0) -> list[tuple[float, float]]:
    """按固定窗口切成 [t0, t1)，左闭右开。"""
    result: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + size, duration)
        result.append((round(start, 3), round(end, 3)))
        start = end
    return result


def frame_at(
    video: Path,
    t: float,
    dst: Path,
    *,
    ffmpeg: str = "",
    png: bool = False,
    width: int = 896,
) -> dict:
    """抽单帧，返回 {t, file, w, h, ok}。

    `-ss` 放在 `-i` 之后是精确 seek，代价是解码到目标位置；单帧仍在几十毫秒级。
    """
    exe = _bin(ffmpeg, DEFAULT_FFMPEG)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 只缩小不放大：竖屏 720x1280 放大到 896 宽会让 OCR 与视觉都变慢
    scale = f"scale='min({width},iw)':-2"
    # -ss 放在 -i 之前是快速 seek：从最近关键帧解码，长视频里便宜得多
    cmd = [exe, "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(video),
           "-frames:v", "1", "-vf", scale]
    if png:
        cmd += [str(dst)]
    else:
        cmd += ["-q:v", "2", str(dst)]
    run(cmd, timeout=120)
    exists = dst.is_file() and dst.stat().st_size > 0
    return {
        "t": round(t, 3),
        "requested_t": round(t, 3),
        "file": dst.name,
        "ok": exists,
        "bytes": dst.stat().st_size if exists else 0,
    }


def probe_tags(video: Path) -> dict:
    """读容器级标签（抖音的 vid 藏在 comment 里，可当视频指纹）。"""
    try:
        result = run([
            DEFAULT_FFPROBE, "-v", "error",
            "-show_entries", "format_tags", "-of", "json", str(video),
        ], timeout=30)
        data = json.loads(result.stdout.decode("utf-8", "ignore") or "{}")
        tags = (data.get("format") or {}).get("tags") or {}
        return {str(k).lower(): str(v) for k, v in tags.items()}
    except Exception:  # noqa: BLE001
        return {}
