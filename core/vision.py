"""视觉端：llama-server + 逐帧结构化标注。

按 ARCHITECTURE §12.3/§12.4：
  · 必须带 --image-min-tokens 1024
  · 用 JSON Schema 约束输出
  · 系统提示词要求「只输出 JSON、不做分析、中文」
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

SYSTEM_PROMPT = (
    "你是视频画面标注器。只输出 JSON，不要分析过程、不要 Markdown、不要解释。"
    "所有字段值必须用中文（专有名词、数字、英文缩写可保留原文）。"
    "看不清就写进 uncertain，不要猜测。"
)

USER_PROMPT = (
    "看这一帧，输出 JSON：\n"
    '{"scene":"一句话描述画面","subjects":["人物/主体及其外观"],'
    '"action":"人物正在做什么、彼此什么关系","mood":"整体情绪或氛围",'
    '"on_screen_text":["画面里的文字，逐条原样照抄"],"uncertain":[]}\n'
    "没有文字时 on_screen_text 为空数组。"
)

FRAME_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string"},
        "subjects": {"type": "array", "items": {"type": "string"}},
        "action": {"type": "string"},
        "mood": {"type": "string"},
        "on_screen_text": {"type": "array", "items": {"type": "string"}},
        "uncertain": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["scene", "subjects", "action", "mood", "on_screen_text", "uncertain"],
}

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class VisionService:
    """一个最小的 llama-server 生命周期 + 单帧标注客户端。"""

    def __init__(
        self,
        *,
        server_exe: str,
        model_path: str,
        mmproj_path: str,
        port: int = 8090,
        ctx_size: int = 8192,
        n_gpu_layers: int = 99,
        timeout: float = 180.0,
        boot_timeout: float = 120.0,
        threads: int = 6,
    ) -> None:
        self.server_exe = server_exe
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.port = port
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.timeout = timeout
        self.boot_timeout = boot_timeout
        self.threads = max(1, int(threads))
        self._proc: subprocess.Popen | None = None
        self._started_at = 0.0

    # ---------------- 服务生命周期 ----------------
    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def healthy(self) -> bool:
        try:
            with _OPENER.open(f"{self.base}/health", timeout=3) as resp:
                return json.loads(resp.read().decode()).get("status") == "ok"
        except Exception:  # noqa: BLE001
            return False

    def ensure_server(self) -> bool:
        """确保服务在跑；返回 True 表示本次新启动。"""
        if self.healthy():
            return False
        self._proc = subprocess.Popen(
            [
                self.server_exe,
                "-m", self.model_path,
                "--mmproj", self.mmproj_path,
                "--host", "127.0.0.1",
                "--port", str(self.port),
                "--ctx-size", str(self.ctx_size),
                "--parallel", "1",
                "--threads", str(self.threads),
                "-ngl", str(self.n_gpu_layers),
                "--image-min-tokens", "1024",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + self.boot_timeout
        while time.time() < deadline:
            if self.healthy():
                self._started_at = time.time()
                return True
            if self._proc.poll() is not None:
                raise RuntimeError(f"视觉服务启动失败，退出码 {self._proc.returncode}")
            time.sleep(1.5)
        raise RuntimeError("视觉服务启动超时")

    def shutdown(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    # ---------------- 单帧标注 ----------------
    @staticmethod
    def _salvage(raw: str) -> dict:
        """模型偶发吐残破 JSON，用正则把关键字段捞回来。"""
        import re as _re

        def pick(field: str) -> str:
            match = _re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
            return match.group(1) if match else ""

        scene = pick("scene")
        if not scene:
            return {}
        return {
            "scene": scene,
            "action": pick("action"),
            "mood": pick("mood"),
            "subjects": [],
            "on_screen_text": [],
            "uncertain": [],
        }

    def _post_with_retry(self, payload: dict) -> dict:
        """视觉服务被意外关闭时，重启后重试一次。"""
        try:
            return self._post(payload)
        except Exception as exc:  # noqa: BLE001
            text = str(exc)
            if "10061" in text or "refused" in text.lower() or "Connection" in text:
                logger.warning("视觉服务无响应，尝试重启后重试")
                self._proc = None
                self.ensure_server(force=True) if "force" in self.ensure_server.__code__.co_varnames else self.ensure_server()
                return self._post(payload)
            raise

    def describe(self, image_path: Path, t: float) -> dict:
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        payload = {
            "model": "local",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": 400,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": FRAME_SCHEMA},
        }
        started = time.time()
        try:
            data = self._post_with_retry(payload)
        except urllib.error.HTTPError:
            payload.pop("response_format", None)
            data = self._post_with_retry(payload)
        elapsed = time.time() - started

        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        parsed = _parse_json(text)
        parsed["t"] = round(t, 3)
        parsed["seconds"] = round(elapsed, 2)
        return parsed

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with _OPENER.open(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())


def _parse_json(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?|```$", "", raw).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        data = {}
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = {}
    return {
        "scene": str(data.get("scene") or raw[:200]),
        "action": str(data.get("action") or ""),
        "mood": str(data.get("mood") or ""),
        "subjects": [str(s) for s in (data.get("subjects") or []) if str(s).strip()],
        "on_screen_text": [
            str(s).strip() for s in (data.get("on_screen_text") or []) if str(s).strip()
        ],
        "uncertain": [str(s) for s in (data.get("uncertain") or []) if str(s).strip()],
    }
