"""解析器缓存对接：@机器人 + 链接 → 找到 parser 下载好的视频文件。

parser 的命名规则（core/utils.py:generate_file_name）：
    md5(url)[:16] + 后缀（URL 里没有后缀时用默认 ".mp4"）
缓存目录：data/plugin_data/astrbot_plugin_parser/cache
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from urllib.parse import urlparse

CACHE_DIR = Path.home() / ".astrbot" / "data" / "plugin_data" / "astrbot_plugin_parser" / "cache"
URL_RE = re.compile(r"https?://[^\s，。！？、）)】\]\"'<>]+")

SUFFIXES = (".mp4", ".mkv", ".mov", ".webm", ".m4a", ".mp3")


def file_stem(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]


def candidate_names(url: str) -> list[str]:
    path_suffix = Path(urlparse(url).path).suffix.lower()
    suffixes = [path_suffix] if path_suffix else []
    suffixes += [s for s in SUFFIXES if s not in suffixes]
    return [file_stem(url) + suffix for suffix in suffixes]


def find_cached(url: str, *, cache_dir: Path | None = None) -> Path | None:
    base = cache_dir or CACHE_DIR
    if not base.is_dir():
        return None
    for name in candidate_names(url):
        path = base / name
        if path.is_file() and path.stat().st_size > 0:
            return path
    # 宽松匹配：某些下载器会加 _lc 之类的后缀
    stem = file_stem(url)
    for path in base.glob(f"{stem}*"):
        if path.is_file() and path.suffix.lower() in SUFFIXES and path.stat().st_size > 0:
            return path
    return None


def wait_cached(
    url: str,
    *,
    cache_dir: Path | None = None,
    timeout: float = 30.0,
    interval: float = 0.8,
) -> Path | None:
    """等 parser 把文件下好（它和我们在同一条消息上并行跑）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        found = find_cached(url, cache_dir=cache_dir)
        if found:
            return found
        time.sleep(interval)
    return None


def newest_video(cache_dir: Path | None = None, max_age: float = 120.0) -> Path | None:
    """兜底：没有链接可算时，取最近下载的视频（2 分钟内）。"""
    base = cache_dir or CACHE_DIR
    if not base.is_dir():
        return None
    videos = [
        p for p in base.iterdir()
        if p.is_file() and p.suffix.lower() in SUFFIXES and p.stat().st_size > 0
    ]
    if not videos:
        return None
    newest = max(videos, key=lambda p: p.stat().st_mtime)
    if time.time() - newest.stat().st_mtime > max_age:
        return None
    return newest

def wait_new_video(
    since: float,
    *,
    cache_dir: Path | None = None,
    timeout: float = 60.0,
    interval: float = 0.8,
    min_bytes: int = 200_000,
) -> Path | None:
    """等本次消息之后新出现的视频文件（哈希对不上时的兜底）。"""
    base = cache_dir or CACHE_DIR
    if not base.is_dir():
        return None
    deadline = time.time() + timeout
    while time.time() < deadline:
        fresh = [
            p
            for p in base.iterdir()
            if p.is_file()
            and p.suffix.lower() in SUFFIXES
            and p.stat().st_size >= min_bytes
            and p.stat().st_mtime >= since
        ]
        if fresh:
            # 同一时刻可能落了多个，取最大的（视频通常比音频大）
            return max(fresh, key=lambda p: p.stat().st_size)
        time.sleep(interval)
    return None


def recent_video(
    since: float,
    *,
    cache_dir: Path | None = None,
    min_bytes: int = 200_000,
    exclude: set[str] | None = None,
) -> Path | None:
    """取 since 之后落盘的最新视频（用于"解析器比我们先跑完"的情况）。"""
    base = cache_dir or CACHE_DIR
    if not base.is_dir():
        return None
    skip = exclude or set()
    candidates = [
        p
        for p in base.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUFFIXES
        and p.stat().st_size >= min_bytes
        and p.stat().st_mtime >= since
        and str(p) not in skip
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


TAG_RE = re.compile(r"#\s*([^#\s]{1,30})")
NOISE_RE = re.compile(
    r"(复制此链接|打开Dou音搜索|直接观看视频|复制打开抖音|看看作品|看一看|"
    r"复制链接|打开抖音|抖音搜索|—|\d+\.\d+\s*\w+:/|v@\S+|:\d+pm|\d{2}/\d{2})"
)


def parse_share_text(text: str) -> dict:
    """从分享文案里抽标题与话题标签（都不强求，取不到就空）。"""
    if not text:
        return {"title": "", "tags": []}

    tags = [
        t.strip()
        for t in TAG_RE.findall(text)
        if t.strip() and _looks_meaningful(t.strip())
    ]
    # 去重保序
    seen: set[str] = set()
    unique_tags = [t for t in tags if not (t in seen or seen.add(t))]

    # 去掉链接、标签、噪声后剩下的最长一段当标题
    body = URL_RE.sub(" ", text)
    body = TAG_RE.sub(" ", body)
    body = NOISE_RE.sub(" ", body)
    chunks = [c.strip(" ，,、:：;；-—") for c in re.split(r"\s{1,}|\n", body)]
    chunks = [c for c in chunks if len(c) >= 4 and _looks_meaningful(c)]
    title = max(chunks, key=len) if chunks else ""
    if not _looks_meaningful(title):
        title = ""
    return {"title": title[:80], "tags": unique_tags[:10]}


def _looks_meaningful(text: str) -> bool:
    """分享文案里夹杂的乱码/口令不算标题或标签。"""
    if not text:
        return False
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    if cjk >= 2:
        return True
    # 纯英文标签允许（如 #vlog），但要求至少 3 个字母且不含口令符号
    letters = sum(1 for ch in text if ch.isalpha())
    return letters >= 3 and not any(ch in text for ch in ":/@.\\|")
