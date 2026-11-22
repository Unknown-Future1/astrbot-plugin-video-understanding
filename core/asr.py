"""语音端：SenseVoice（sherpa-onnx）全量转写 + 句级时间戳。

参考 asr/transcribe.py 的实现，但对外只暴露证据契约需要的形状：
  entries: [{id, t0, t1, text, words, emotion}]

分段策略（ARCHITECTURE §3.2）：超过 chunk_seconds 的音频在内部切块，
块间留 overlap 秒重叠，转写后按时间戳去重合并——对外仍是全时段连续结果。
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import logging

import numpy as np

logger = logging.getLogger("video.asr")

SENT_END = "。！？!?"
SENT_SOFT = "，；,;"
SOFT_SPLIT_LEN = 14
SPECIAL = re.compile(r"<\|[^|]*\|>")
VALID_TEXT = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
FUNC_WORDS = re.compile(r"[的了是我你他她它不在有这那就都也还和吧呢啊嘛哦么什怎吗]")
CJK = re.compile(r"[\u4e00-\u9fff]")

PHANTOM = {
    "抖音", "谢谢观看", "请点赞", "关注", "字幕", "本视频", "视频", "以上",
    "点赞", "转发", "分享", "音乐", "谢谢大家",
}

EMO_MAP = {
    "HAPPY": "开心", "SAD": "悲伤", "ANGRY": "生气", "NEUTRAL": "平静",
    "FEARFUL": "害怕", "DISGUSTED": "厌恶", "SURPRISED": "惊讶",
    "EMO_UNKNOWN": "未知", "OTHER": "其他",
}

try:  # 可选依赖：繁转简
    import opencc

    _CC = opencc.OpenCC("t2s")
except Exception:  # noqa: BLE001
    _CC = None


def to_simplified(text: str) -> str:
    if _CC is None or not text:
        return text
    try:
        return _CC.convert(text)
    except Exception:  # noqa: BLE001
        return text


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(str(path)) as f:
        sr = f.getframerate()
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        if f.getnchannels() > 1:
            data = data.reshape(-1, f.getnchannels()).mean(axis=1).astype(np.int16)
    return data.astype(np.float32) / 32768.0, sr


def quality(text: str, duration: float) -> tuple[float, str]:
    """粗判一句转写是否可信（与 ASR 插件同一套启发式）。"""
    chars = CJK.findall(text)
    n = len(chars)
    if n == 0:
        return 0.0, "empty"
    cps = n / max(duration, 0.3)
    speed = 1.0 if 1.5 <= cps <= 9.0 else 0.25
    uniq = len(set(chars)) / n
    diversity = 1.0 if uniq >= 0.42 else 0.4
    func = len(FUNC_WORDS.findall(text)) / n
    function_score = 1.0 if func >= 0.03 else 0.55
    return round(speed * diversity * function_score, 3), f"speed={cps:.1f}cps uniq={uniq:.2f} func={func:.2f}"


def split_sentences(
    tokens: list[str], stamps: list[float]
) -> list[tuple[float, float, str]]:
    """按标点切句，时间取 token 时间戳（左闭右开由调用方保证）。"""
    sentences: list[tuple[float, float, str]] = []
    buffer: list[str] = []
    start: float | None = None
    last = 0.0
    for token, ts in zip(tokens, stamps):
        clean = SPECIAL.sub("", token)
        if not clean:
            continue
        last = ts
        if start is None:
            start = ts
        buffer.append(clean)
        joined = "".join(buffer)
        hard = clean in SENT_END
        soft = clean in SENT_SOFT and len(joined) >= SOFT_SPLIT_LEN
        if hard or soft:
            text = to_simplified(joined.strip())
            if VALID_TEXT.search(text):
                sentences.append((round(start, 3), round(ts, 3), text))
            buffer, start = [], None
    if buffer and start is not None:
        text = to_simplified("".join(buffer).strip())
        if VALID_TEXT.search(text):
            sentences.append((round(start, 3), round(last, 3), text))
    return sentences


class AsrService:
    def __init__(
        self,
        *,
        model_dir: str,
        num_threads: int = 6,
        chunk_seconds: float = 20.0,
        overlap: float = 0.5,
        keep_loaded: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.num_threads = max(1, int(num_threads))
        self.chunk_seconds = max(10.0, float(chunk_seconds))
        self.overlap = max(0.0, float(overlap))
        self.keep_loaded = keep_loaded
        self._rec = None
        self.engine = "sensevoice"

    # ---------------- 模型 ----------------
    def available(self) -> tuple[bool, str]:
        model = self.model_dir / "model.int8.onnx"
        tokens = self.model_dir / "tokens.txt"
        missing = [str(p) for p in (model, tokens) if not p.is_file()]
        if missing:
            return False, "SenseVoice 模型文件缺失: " + ", ".join(missing)
        return True, ""

    def _ensure(self):
        if self._rec is not None:
            return self._rec
        ok, err = self.available()
        if not ok:
            raise RuntimeError(err)
        import sherpa_onnx

        self._rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(self.model_dir / "model.int8.onnx"),
            tokens=str(self.model_dir / "tokens.txt"),
            num_threads=self.num_threads,
            use_itn=True,
            language="zh",
            debug=False,
        )
        return self._rec

    def unload(self) -> None:
        self._rec = None

    # ---------------- 转写 ----------------
    def _run_chunk(self, samples: np.ndarray) -> dict:
        rec = self._ensure()
        stream = rec.create_stream()
        stream.accept_waveform(16000, samples.astype(np.float32, copy=False))
        rec.decode_stream(stream)
        result = stream.result
        raw_emo = (getattr(result, "emotion", "") or "").strip("<>|").upper()
        return {
            "tokens": list(getattr(result, "tokens", []) or []),
            "timestamps": list(getattr(result, "timestamps", []) or []),
            "emotion": EMO_MAP.get(raw_emo) if raw_emo else None,
            "event": (getattr(result, "event", "") or "").strip("<>|") or None,
        }

    # ---------------- VAD ----------------
    def _vad_segments(self, samples: np.ndarray, sr: int) -> list[tuple[float, np.ndarray]]:
        """用 silero-vad 切出人声段；不可用时返回空列表。"""
        model = Path(self.model_dir).parent / "silero_vad.onnx"
        if not model.is_file():
            return []
        try:
            import sherpa_onnx

            cfg = sherpa_onnx.VadModelConfig()
            cfg.silero_vad.model = str(model)
            cfg.silero_vad.threshold = 0.35
            cfg.silero_vad.min_silence_duration = 0.3
            cfg.silero_vad.min_speech_duration = 0.2
            cfg.sample_rate = sr
            vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"VAD 初始化失败，回退固定分块: {exc}")
            return []

        segments: list[tuple[float, np.ndarray]] = []
        step = sr
        for i in range(0, len(samples), step):
            vad.accept_waveform(samples[i : i + step])
            while not vad.empty():
                seg = vad.front
                segments.append((seg.start / sr, np.array(seg.samples, dtype=np.float32)))
                vad.pop()
        vad.flush()
        while not vad.empty():
            seg = vad.front
            segments.append((seg.start / sr, np.array(seg.samples, dtype=np.float32)))
            vad.pop()
        return segments

    def transcribe(self, wav_path: Path) -> dict:
        started = time.time()
        samples, sr = load_wav(wav_path)
        if sr != 16000:
            raise RuntimeError(f"音频需要 16 kHz，实际 {sr}")
        duration = len(samples) / sr

        sentences: list[dict] = []
        emotions: list[str] = []
        events: list[str] = []

        vad_segments = self._vad_segments(samples, sr)
        if vad_segments:
            logger.info(f"VAD 切出 {len(vad_segments)} 段人声")
            for start_sec, piece in vad_segments:
                if len(piece) < sr * 0.3:
                    continue
                info = self._run_chunk(piece)
                if info["emotion"]:
                    emotions.append(info["emotion"])
                if info.get("event"):
                    events.append(info["event"])
                for t0, t1, text in split_sentences(info["tokens"], info["timestamps"]):
                    stripped = text.strip("，,。！？!?；; ")
                    if not stripped or stripped in PHANTOM:
                        continue
                    sentences.append(
                        {
                            "t0": round(start_sec + t0, 3),
                            "t1": round(start_sec + t1, 3),
                            "text": text,
                        }
                    )
            merged_vad = sorted(sentences, key=lambda x: x["t0"])
            sentences = merged_vad
            chunk = int(self.chunk_seconds * sr)
            if True:  # 跳过下面的固定分块
                merged = sentences
                for index, item in enumerate(merged):
                    item["id"] = f"s{index + 1}"
                    item["alignment"] = "sentence"
                    item["words"] = []
                    score, reason = quality(item["text"], max(0.3, item["t1"] - item["t0"]))
                    item["quality"] = score
                    item["quality_reason"] = reason
                    item["suspect"] = score < 0.5
                # 可疑句重跑：按 6 秒窗口再识别一次，取更可信的
                for item in merged:
                    if not item["suspect"]:
                        continue
                    start = max(0.0, item["t0"] - 0.2)
                    end = min(duration, item["t1"] + 0.2)
                    piece = samples[int(start * sr) : int(end * sr)]
                    if len(piece) < sr * 0.3:
                        continue
                    info = self._run_chunk(piece)
                    for t0, t1, text in split_sentences(info["tokens"], info["timestamps"]):
                        new_score, _ = quality(text, max(0.3, t1 - t0))
                        if new_score > item.get("quality", 0):
                            item["original_text"] = item["text"]
                            item["text"] = text
                            item["quality"] = new_score
                            item["suspect"] = new_score < 0.5
                            break
                return {
                    "status": "ok" if merged else "silent",
                    "has_speech": bool(merged),
                    "engine": self.engine,
                    "duration": round(duration, 3),
                    "entries": merged,
                    "emotion": _dominant(emotions),
                    "events": sorted(set(events)),
                    "seconds": round(time.time() - started, 2),
                }

        chunk = int(self.chunk_seconds * sr)
        step = max(1, int((self.chunk_seconds - self.overlap) * sr))

        if len(samples) <= chunk:
            pieces = [(0, samples)]
        else:
            pieces = []
            pos = 0
            while pos < len(samples):
                pieces.append((pos, samples[pos : pos + chunk]))
                if pos + chunk >= len(samples):
                    break
                pos += step

        for offset, piece in pieces:
            if len(piece) < sr * 0.3:
                continue
            info = self._run_chunk(piece)
            if info["emotion"]:
                emotions.append(info["emotion"])
            if info.get("event"):
                events.append(info["event"])
            base = offset / sr
            for t0, t1, text in split_sentences(info["tokens"], info["timestamps"]):
                stripped = text.strip("，,。！？!?；; ")
                if not stripped or stripped in PHANTOM:
                    continue
                sentences.append(
                    {"t0": round(base + t0, 3), "t1": round(base + t1, 3), "text": text}
                )

        # 分段重叠处去重：同文本且时间相近只留更早的一条
        merged: list[dict] = []
        for item in sorted(sentences, key=lambda x: x["t0"]):
            duplicate = any(
                prev["text"].strip("，,。") == item["text"].strip("，,。")
                and abs(prev["t0"] - item["t0"]) < self.overlap + 0.3
                for prev in merged[-3:]
            )
            if not duplicate:
                merged.append(item)

        for index, item in enumerate(merged):
            item["id"] = f"s{index + 1}"
            item["alignment"] = "sentence"
            item["words"] = []
            score, reason = quality(item["text"], max(0.3, item["t1"] - item["t0"]))
            item["quality"] = score
            item["quality_reason"] = reason
            item["suspect"] = score < 0.5

        return {
            # 有音轨但没人声（纯音乐）= silent；调用方在无音轨时直接给 not_present
            "status": "ok" if merged else "silent",
            "has_speech": bool(merged),
            "engine": self.engine,
            "duration": round(duration, 3),
            "entries": merged,
            "emotion": _dominant(emotions),
            "events": sorted(set(events)),
            "seconds": round(time.time() - started, 2),
        }


def _dominant(items: list[str]) -> str | None:
    if not items:
        return None
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return max(counts, key=counts.get)
