"""OCR 后处理：过滤噪声、标记水印、合并同高度行。"""

from __future__ import annotations

import re
from pathlib import Path

CJK = re.compile(r"[\u4e00-\u9fff]")
ALNUM = re.compile(r"[A-Za-z0-9]")

WATERMARK_HINTS = (
    "抖音", "抖音号", "搜索", "截图保存", "来抖音", "大家都在", "扫一扫",
    "关注", "点赞", "评论", "分享", "转发", "剪映",
)


def _is_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    # 单字符、纯符号、单字母图标误识别
    if len(stripped) == 1 and (not CJK.match(stripped) or ALNUM.match(stripped)):
        return True
    if not CJK.search(stripped) and not re.search(r"[A-Za-z0-9]{2,}", stripped):
        return True
    return False


def clean(lines: list[dict], image_height: int = 0) -> tuple[list[dict], list[dict]]:
    """返回 (正文行, 水印行)，两者都按 y 排序。"""
    body: list[dict] = []
    marks: list[dict] = []
    for line in lines:
        text = str(line.get("text") or "").strip()
        if _is_noise(text):
            continue
        item = {**line, "text": text}
        y_ratio = (line.get("y", 0) / image_height) if image_height else 0.0
        item["y_ratio"] = round(y_ratio, 3)
        # 只有"很靠底 + 像平台标识"才算水印；纯字幕即使靠底也当正文
        low = (line.get("y", 0) + line.get("h", 0)) / image_height if image_height else 0.0
        looks_platform = bool(
            re.search(r"抖音|快手|小红书|B站|bilibili|douyin|kuaishou|号|@|搜索", text, re.I)
        )
        if any(hint in text for hint in WATERMARK_HINTS) or (low >= 0.96 and looks_platform):
            item["kind"] = "watermark"
            marks.append(item)
        else:
            item["kind"] = "body"
            body.append(item)
    body.sort(key=lambda item: (item.get("y", 0), item.get("x", 0)))
    marks.sort(key=lambda item: (item.get("y", 0), item.get("x", 0)))
    return body, marks


def read_clean(service, image_path: Path) -> dict:
    """调 OcrService.read 之后再清洗。"""
    raw = service.read(image_path)
    height = 0
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            height = im.height
    except Exception:  # noqa: BLE001
        height = 0
    body, marks = clean(raw.get("lines") or [], height)
    return {
        "body": body,
        "watermarks": marks,
        "text": " | ".join(item["text"] for item in body),
        "seconds": raw.get("seconds", 0.0),
    }
