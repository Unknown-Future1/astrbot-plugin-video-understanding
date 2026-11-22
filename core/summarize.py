"""成文：先给确定性事实（L1），再让本地模型写一段摘要。

设计取舍（按你的选择）：
  · 不路由模型，只用 Qwen3.5-4B
  · 插件自己完成统合，交给外部大模型的是"事实 + 摘要"，不是原始证据堆
  · 模型只负责组织语言，数字/时间/文字一律来自规则层，禁止改写
"""

from __future__ import annotations

import json
import re

MAX_LINES = 40

SYSTEM = (
    "你在帮群友看一段视频。根据下面给到的镜头、人物、动作、字幕和语音，"
    "讲清楚这段视频在演什么、想表达什么情绪或笑点。"
    "画面与字幕是事实，可以合理推测人物关系和剧情走向，但推测要说得像推测。"
    "不要逐条念时间戳，不要罗列数字清单；像跟朋友分享一个有意思的视频那样说，"
    "180~260 字。听不懂的语音就带过，不要编。"
)


def build_facts(bundle: dict) -> dict:
    """把证据压成可核对的事实层。"""
    speech = bundle["evidence"]["speech_text"]
    visual = bundle["evidence"]["visual_text"]
    persistent = bundle["evidence"].get("persistent_text") or []
    conflicts = bundle["evidence"].get("conflicts") or []

    numbers = _extract_numbers(
        [item["text"] for item in speech],
        [t for item in visual for t in (item.get("on_screen_text") or [])],
    )
    timeline = []
    for item in speech[:MAX_LINES]:
        timeline.append(
            {"t": item["t0"], "speech": item["text"], "suspect": bool(item.get("suspect"))}
        )
    for item in visual:
        texts = " / ".join(item.get("on_screen_text") or [])
        if texts:
            timeline.append({"t": item["t0"], "screen": texts})
    timeline.sort(key=lambda x: x["t"])

    return {
        "title": [item["text"] for item in persistent],
        "numbers": numbers,
        "timeline": timeline,
        "conflicts": [
            {"t": c.get("t"), "a": c.get("a"), "b": c.get("b"), "note": c.get("note")}
            for c in conflicts
        ],
        "uncertain": bundle.get("uncertainty") or [],
        "corrections": bundle.get("corrections") or [],
        "share_title": (bundle.get("source_meta") or {}).get("title", ""),
        "tags": (bundle.get("source_meta") or {}).get("tags") or [],
        "audio_events": bundle["channels"]["speech"].get("events") or [],
        "scenes": _scenes(visual),
        "subtitles": _subtitles(visual),
        "stats": {
            "duration": bundle["video"]["duration"],
            "speech_lines": len(speech),
            "screen_frames": len(visual),
            "profile": (bundle.get("sampling") or {}).get("profile"),
        },
    }


def _scenes(visual: list[dict]) -> list[dict]:
    out = []
    for item in visual:
        out.append(
            {
                "t": item["t0"],
                "scene": item.get("text", ""),
                "action": item.get("action", ""),
                "mood": item.get("mood", ""),
                "subjects": item.get("subjects") or [],
            }
        )
    return out


def _subtitles(visual: list[dict]) -> list[dict]:
    out = []
    for item in visual:
        for text in item.get("on_screen_text") or []:
            out.append({"t": item["t0"], "text": text})
    return out


def facts_text(facts: dict) -> str:
    """把事实层压成给模型看的纯文本。"""
    parts = [f"视频时长 {facts['stats']['duration']}s，"
             f"语音 {facts['stats']['speech_lines']} 句，画面 {facts['stats']['screen_frames']} 帧。"]
    if facts.get("share_title"):
        parts.append("分享标题（作者写的）：" + facts["share_title"])
    if facts.get("tags"):
        parts.append("话题标签：" + "、".join(facts["tags"]))
    if facts["title"]:
        parts.append("画面反复出现的文字：" + "；".join(facts["title"][:6]))
    # 数字不直接列出：避免模型照抄成清单；校验时仍会用到
    if facts.get("audio_events"):
        parts.append("音频事件：" + "、".join(facts["audio_events"]))
    parts.append("镜头（按时间）：")
    for item in facts.get("scenes") or []:
        who = "、".join(item.get("subjects") or [])[:40]
        parts.append(
            f"  {item['t']:.1f}s 画面：{item['scene']}"
            + (f"；人物：{who}" if who else "")
            + (f"；动作：{item['action']}" if item.get("action") else "")
            + (f"；氛围：{item['mood']}" if item.get("mood") else "")
        )
    subs = facts.get("subtitles") or []
    if subs:
        parts.append("字幕/画面文字（按出现顺序）：")
        for item in subs[:20]:
            parts.append(f"  {item['t']:.1f}s 「{item['text']}」")
    speech = [
        item for item in (facts.get("timeline") or []) if "speech" in item
    ]
    if speech:
        parts.append("语音：")
        for item in speech[:10]:
            mark = "（疑似听错）" if item.get("suspect") else ""
            parts.append(f"  {item['t']:.1f}s {mark}{item['speech']}")
    if facts["conflicts"]:
        parts.append("冲突（原样保留，不要自行裁定）：")
        for c in facts["conflicts"][:6]:
            parts.append(f"  {c['t']}s {c.get('a')} ↔ {c.get('b')}（{c.get('note')}）")
    if facts["uncertain"]:
        parts.append("不确定：" + "；".join(facts["uncertain"][:5]))
    return "\n".join(parts)


MAX_FACTS_CHARS = 2600


def narrate(service, facts: dict) -> str:
    """调用本地模型写摘要；失败返回空串（调用方回退到事实层）。"""
    text = facts_text(facts)
    if len(text) > MAX_FACTS_CHARS:
        text = text[:MAX_FACTS_CHARS] + "\n（材料过长，已截断）"
    payload = {
        "model": "local",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
        "max_tokens": 320,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        data = service._post(payload)
    except Exception:  # noqa: BLE001
        return ""
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return (text or "").strip()


def build_digest(bundle: dict, service=None) -> dict:
    facts = build_facts(bundle)
    digest = {"facts": facts, "summary": "", "verified": True, "unknown_numbers": []}
    empty = not facts["timeline"] and not facts["title"]
    if service is not None and not empty:
        digest["summary"] = narrate(service, facts)
        unknown = verify_numbers(digest["summary"], facts) if digest["summary"] else []
        if unknown:
            digest["verified"] = False
            digest["unknown_numbers"] = unknown
    return digest


def render_text(digest: dict) -> str:
    """给外部大模型看的最终交付文本。"""
    lines = []
    if digest["summary"]:
        lines.append("【摘要】")
        lines.append(digest["summary"])
        lines.append("")
    facts = digest["facts"]
    lines.append("【事实】")
    lines.append(facts_text(facts))
    if facts["conflicts"]:
        lines.append("")
        lines.append("【需要判断的冲突】")
        for c in facts["conflicts"]:
            lines.append(f"- {c['t']}s：{c.get('a')} ↔ {c.get('b')}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(render_text(build_digest(data)))

def verify_numbers(summary: str, facts: dict) -> list[str]:
    """返回摘要里出现、但事实层没有的数字（含日期片段）。"""
    if not summary:
        return []
    allowed = set(facts.get("numbers") or [])
    for item in facts.get("timeline") or []:
        for token in re.findall(r"\d+(?:\.\d+)?", item.get("speech", "") + item.get("screen", "")):
            allowed.add(token)
    # 时间戳、时长、统计数也属于"事实里出现过"的数字
    for extra in (
        str(facts.get("stats", {}).get("duration", "")),
        *(str(item.get("t", "")) for item in facts.get("timeline") or []),
        *(str(item.get("t", "")) for item in facts.get("conflicts") or []),
        str(facts.get("stats", {}).get("speech_lines", "")),
        str(facts.get("stats", {}).get("screen_frames", "")),
    ):
        for token in re.findall(r"\d+(?:\.\d+)?", extra):
            allowed.add(token)
    times: list[float] = []
    for item in facts.get("timeline") or []:
        try:
            times.append(float(item.get("t", 0)))
        except (TypeError, ValueError):
            continue
    for item in facts.get("conflicts") or []:
        try:
            times.append(float(item.get("t", 0)))
        except (TypeError, ValueError):
            continue
    duration = float((facts.get("stats") or {}).get("duration") or 0)

    unknown = []
    for token in re.findall(r"\d+(?:\.\d+)?", summary):
        if token in allowed:
            continue
        # 小数且落在某个时间点 ±0.3 秒内 → 视为在描述时间，不算编造
        if "." in token:
            value = float(token)
            if any(abs(value - t) <= 0.3 for t in times) or abs(value - duration) <= 0.3:
                continue
        unknown.append(token)
    return sorted(set(unknown))


def _plausible_number(text: str) -> bool:
    """过滤 OCR/ASR 拼出来的怪数字（1~4 位整数 + 最多 1 位小数）。"""
    import re as _re

    if not _re.fullmatch(r"\d{1,4}(?:\.\d)?", text):
        return False
    if len(text) > 1 and text.startswith("0") and "." not in text:
        return False
    if text.endswith("."):
        return False
    return True


_UNIT = r"(?:w|W|fps|帧|%|度|元|尼特|°C|℃|秒|年|月|日|次|倍|nit|Hz|hz)"
_UNIT_NUM = re.compile(r"(\d{1,4}(?:\.\d)?)\s*" + _UNIT)
_DECIMAL = re.compile(r"\b(\d{2,3}\.\d)\b")


def _extract_numbers(speech_texts: list[str], screen_texts: list[str]) -> list[str]:
    """语音里的数字全收；画面里的只收"带单位"或"两位整数+一位小数"的。"""
    found: set[str] = set()
    for text in speech_texts:
        for token in re.findall(r"\d+(?:\.\d+)?", text):
            if _plausible_number(token):
                found.add(token)
    for text in screen_texts:
        for token in _UNIT_NUM.findall(text):
            found.add(token)
        for token in _DECIMAL.findall(text):
            found.add(token)
    return sorted(found)
