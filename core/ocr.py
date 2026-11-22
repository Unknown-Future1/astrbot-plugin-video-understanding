"""OCR 端：RapidOCR（PP-OCRv4 / ONNX / CPU）。

依赖是可选导入：装了就用，没装就返回不可用，让视觉路退回「VLM 读字」。
"""

from __future__ import annotations

import time
from pathlib import Path


class OcrService:
    def __init__(
        self,
        *,
        enabled: bool = True,
        min_conf: float = 0.5,
        num_threads: int | None = None,
    ) -> None:
        self.enabled = enabled
        self.min_conf = min_conf
        from .settings import default_ocr_threads

        self.num_threads = max(1, int(num_threads or default_ocr_threads()))
        self._engine = None
        self._error = ""

    def available(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "OCR 已关闭"
        try:
            import rapidocr_onnxruntime  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, f"缺少 rapidocr_onnxruntime: {exc}"
        return True, ""

    def _ensure(self):
        if self._engine is not None:
            return self._engine
        ok, err = self.available()
        if not ok:
            raise RuntimeError(err)
        import os

        # onnxruntime 认这两个环境变量；RapidOCR 也支持显式传参，双保险
        os.environ.setdefault("OMP_NUM_THREADS", str(self.num_threads))
        os.environ.setdefault("ORT_NUM_THREADS", str(self.num_threads))
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR(
            intra_op_num_threads=self.num_threads,
            inter_op_num_threads=1,
        )
        return self._engine

    def read(self, image_path: Path) -> dict:
        """返回 {lines:[{text, conf, box}], text, seconds}。"""
        started = time.time()
        engine = self._ensure()
        result, _ = engine(str(image_path))
        lines = []
        for item in result or []:
            box, text, conf = item[0], item[1], float(item[2])
            if conf < self.min_conf or not str(text).strip():
                continue
            ys = [point[1] for point in box]
            xs = [point[0] for point in box]
            lines.append(
                {
                    "text": str(text).strip(),
                    "conf": round(conf, 3),
                    "y": round(sum(ys) / len(ys), 1),
                    "x": round(min(xs), 1),
                }
            )
        lines.sort(key=lambda item: (item["y"], item["x"]))
        return {
            "lines": lines,
            "text": " | ".join(item["text"] for item in lines),
            "seconds": round(time.time() - started, 3),
        }

    def unload(self) -> None:
        self._engine = None
