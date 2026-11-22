"""证据组装：把视觉帧与语音句合成 ARCHITECTURE §4 的 EvidenceBundle。"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timeline(audio_start: float = 0.0) -> dict:
    return {
        "origin": "video_start",
        "unit": "s",
        "precision": 3,
        "ranges": "half_open",
        "audio_start": round(audio_start, 3),
    }


def build(
    *,
    job_id: str,
    info: dict,
    frames: list[dict],
    speech: dict,
    workdir: Path,
    visual_model: str,
) -> dict:
    """frames: [{t, file, ok, scene, subjects, on_screen_text, uncertain, seconds}]"""
    ok_frames = [f for f in frames if f.get("ok")]
    visual_entries = []
    keyframes = []
    uncertainty: list[str] = []

    for index, frame in enumerate(ok_frames, start=1):
        path = workdir / "frames" / frame["file"]
        keyframes.append(
            {
                "t": frame["t"],
                "file": f"frames/{frame['file']}",
                "sha256": sha256_file(path) if path.is_file() else "",
                "w": frame.get("w", 0),
                "h": frame.get("h", 0),
            }
        )
        visual_entries.append(
            {
                "id": f"v{index}",
                "t0": frame["t"],
                "t1": frame["t"],
                "text": frame.get("scene", ""),
                "subjects": frame.get("subjects") or [],
                "on_screen_text": frame.get("on_screen_text") or [],
                "watermarks": frame.get("watermarks") or [],
            }
        )
        for item in frame.get("uncertain") or []:
            uncertainty.append(f"{frame['t']:.1f}s: {item}")

    persistent = _extract_persistent(visual_entries)
    speech_entries = speech.get("entries") or []
    conflicts = _detect_conflicts(visual_entries, speech_entries, persistent)

    channels = {
        "visual": {
            "status": "ok" if visual_entries else "failed",
            "model": visual_model,
            "windows": len(visual_entries),
        },
        "speech": {
            "status": speech.get("status", "failed"),
            "events": speech.get("events") or [],
            "models": [speech.get("engine", "sensevoice")],
            "emotion": speech.get("emotion"),
            "seconds": speech.get("seconds"),
        },
        "frames": {"status": "ok" if keyframes else "failed", "count": len(keyframes)},
    }

    bundle = {
        "job_id": job_id,
        "video": info,
        "channels": channels,
        "timeline": timeline(),
        "evidence": {
            "visual_text": visual_entries,
            "speech_text": speech_entries,
            "persistent_text": persistent,
            "conflicts": conflicts,
        },
        "keyframes": keyframes,
        "uncertainty": uncertainty,
        "legacy": False,
    }
    bundle["merged_index"] = _merged_index(visual_entries, speech_entries)
    return bundle


def _merged_index(visual: list[dict], speech: list[dict]) -> list[dict]:
    items = []
    for entry in visual:
        items.append({"t": entry["t0"], "id": entry["id"], "brief": entry["text"][:30]})
    for entry in speech:
        items.append(
            {"t": entry["t0"], "id": entry["id"], "brief": f"「{entry['text'][:30]}」"}
        )
    return sorted(items, key=lambda item: item["t"])


_NUM = r"\d+(?:\.\d+)?"
import re  # noqa: E402

_NUM_RE = re.compile(_NUM)


def _detect_conflicts(
    visual: list[dict], speech: list[dict], persistent: list[dict] | None = None
) -> list[dict]:
    """两类确定性冲突：数字对不上、关键词近似但不一致（如 道歉/倒欠）。"""
    conflicts: list[dict] = []
    persistent_lines = [item["text"] for item in (persistent or [])]
    for entry in speech:
        window = [
            item for item in visual if abs(item["t0"] - entry["t0"]) <= 3.5
        ]
        screen_lines = [t for item in window for t in (item.get("on_screen_text") or [])]
        screen_lines += persistent_lines
        if not screen_lines:
            continue

        numbers = [n for n in _NUM_RE.findall(entry["text"]) if len(n) >= 2]
        screen_text = " ".join(screen_lines)
        missing = [n for n in numbers if n not in screen_text]
        if missing and numbers:
            conflicts.append(
                {
                    "type": "audio_visual_mismatch",
                    "t": entry["t0"],
                    "note": f"语音数字 {missing} 未出现在画面文字里",
                    "speech_id": entry["id"],
                }
            )

        for fragment in _fragments(entry["text"]):
            for line in screen_lines:
                if _same_sound(fragment, line):
                    conflicts.append(
                        {
                            "type": "asr_disagreement",
                            "t": entry["t0"],
                            "a": fragment,
                            "b": line,
                            "note": "同音异字，语音与画面写法不一致",
                            "speech_id": entry["id"],
                        }
                    )
                    break
                if _edit_distance_one(fragment, line):
                    conflicts.append(
                        {
                            "type": "asr_disagreement",
                            "t": entry["t0"],
                            "a": fragment,
                            "b": line,
                            "note": "语音与画面仅一字之差，需核对",
                            "speech_id": entry["id"],
                        }
                    )
                    break
            else:
                continue
            break
    return conflicts


def _extract_persistent(visual: list[dict], ratio: float = 0.6) -> list[dict]:
    """跨帧重复出现的文字（标题栏/水印）挑出来，只留首次出现时间。"""
    if len(visual) < 2:
        return []
    counts: dict[str, list[float]] = {}
    for entry in visual:
        for text in entry.get("on_screen_text") or []:
            counts.setdefault(text, []).append(entry["t0"])
    threshold = max(2, int(len(visual) * ratio))
    persistent: list[dict] = []
    for text, times in counts.items():
        if len(times) < threshold:
            continue
        # 过滤统计叠字/图标拼出来的假标题：含 / % 、数量字多、过短过长
        digits = sum(1 for ch in text if ch.isdigit())
        if (
            "/" in text
            or "%" in text
            or len(text) < 4
            or len(text) > 20
            or digits / max(1, len(text)) > 0.3
        ):
            continue
        if True:
            persistent.append(
                {"text": text, "frames": len(times), "first_t": round(min(times), 3)}
            )
            for entry in visual:
                entry["on_screen_text"] = [
                    item for item in (entry.get("on_screen_text") or []) if item != text
                ]
    persistent.sort(key=lambda item: item["first_t"])
    return persistent


def _edit_distance_one(a: str, b: str) -> bool:
    """长度相同或差 1，且编辑距离为 1。"""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) == 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    i = j = 0
    skipped = False
    while i < len(short) and j < len(long):
        if short[i] == long[j]:
            i += 1
            j += 1
        elif skipped:
            return False
        else:
            skipped = True
            j += 1
    return True


def _fragments(text: str) -> list[str]:
    import re as _re

    parts = _re.split(r"[，。！？、；,;!?\s]+", text)
    return [p for p in parts if len(p) >= 2]


def _same_sound(a: str, b: str) -> bool:
    """长度相同、拼音相同但字不同 → 同音异字（如 道歉/倒欠）。"""
    if len(a) != len(b) or a == b:
        return False
    try:
        from pypinyin import lazy_pinyin
    except Exception:  # noqa: BLE001
        return False
    pa, pb = lazy_pinyin(a), lazy_pinyin(b)
    return len(pa) == len(pb) and pa == pb
