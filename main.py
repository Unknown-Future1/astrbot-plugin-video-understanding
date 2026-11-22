"""astrbot_plugin_video_understanding —— @机器人 + 视频链接 → 内容解析。

当前版本（0.1.0）只做一条路径：
    群里 @机器人 + 抖音/B站/YouTube 链接
      → 等 astrbot_plugin_parser 把视频下到缓存
      → 场景切点 + OCR + ASR + 视觉 + 本地 4B 成文
      → 把摘要与事实发回群

设置项暂未做面板，全部走默认值（需要改时改插件配置）。
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
import json as _json

from astrbot.api.message_components import Json, Record, Reply, Video
from astrbot.api.star import Context, Star, StarTools, register

from .core import settings, source, summarize
from .core.pipeline import analyze_video

__version__ = "0.1.0"

VIDEO_HOSTS = (
    "douyin.com", "v.douyin.com", "iesdouyin.com",
    "bilibili.com", "b23.tv",
    "youtube.com", "youtu.be",
    "kuaishou.com", "ixigua.com",
    "xiaohongshu.com", "xhslink.com",
    "tiktok.com", "weibo.com", "weibo.cn",
)

URL_RE = re.compile(r"https?://[^\s，。！？、）)】\]\"'<>]+")

# 人格提示词用占位符，运行期再填；日志只打印卡片名
SYSTEM_TEMPLATE = (
    "{persona}\n\n"
    "【补充说明】\n"
    "你刚看完群友发的一条视频。下面「解析结果」是你看完之后的材料，不是别人转述的。\n"
    "请用你自己的口吻，跟大家讲讲这段视频在演什么、想表达什么情绪或笑点：\n"
    "· 像跟朋友分享一个有意思的视频，2~4 句，别写报告\n"
    "· 人物、画面、字幕是事实；人物关系和剧情可以合理推测，但要说成像推测（「像是在…」）\n"
    "· 不要逐条念时间戳、不要罗列字幕清单；只在关键转折处顺口提一句时间\n"
    "· 语音转写可能有同音错字，画面文字更可靠；实在听不懂就说没太听清\n"
    "· 分享文案里的标题和话题标签可以帮你判断人物/主题"
    "（比如 #灭霸 对应画面里的紫色肌肉男），但 #剪映、#AI创作浪潮计划 这类工具/活动标签要忽略\n"
)

DEFAULTS = {
    "llama_dir": r"H:\Deepseek-Harness\Workspace\model\llama\bin-vulkan",
    "model_path": r"H:\Deepseek-Harness\Workspace\model\qwen3.5-4b\Qwen3.5-4B-Q4_K_M.gguf",
    "mmproj_path": r"H:\Deepseek-Harness\Workspace\model\qwen3.5-4b\mmproj-F16.gguf",
    "asr_model_dir": r"H:\Deepseek-Harness\Workspace\asr\models\sensevoice",
    "port": 8090,
    "ocr_threads": 0,
    "vlm_threads": 0,
    "asr_threads": 0,
    "frame_cap": 0,
    "keep_server": True,
    "parse_wait": 60,
    "reply_max_chars": 1200,
    "use_chat_model": True,
    "term_correction": False,
    "parser_cache_dir": "",
    "idle_unload_seconds": 600,
    "vlm_image_width": 640,
}


@register(
    "astrbot_plugin_video_understanding",
    "未知の未来",
    "视频理解：@机器人 + 链接，自动抽帧/OCR/语音识别/成文",
    __version__,
)
class VideoUnderstandingPlugin(Star):
    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self.config = {**DEFAULTS, **(config or {})}
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._last_used = 0.0
        self._busy = False
        self._idle_task: asyncio.Task | None = None
        self._consumed: set[str] = set()
        self._uid_meta: dict[str, dict] = {}
        self._url_map: dict[str, str] = {}

    # ---------------- 触发 ----------------
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_link(self, event: AstrMessageEvent):
        if not self._enabled():
            return
        if not event.is_at_or_wake_command:
            return

        urls = self._collect_urls(event)
        target = self._pick_url(urls) if urls else None

        # 引用视频：不要求有链接文字
        quoted = self._quoted_video(event)
        if not target and quoted is None:
            return
        if quoted is None and target and not self._looks_like_video(target):
            return

        self._started_at = time.time()
        self._meta = source.parse_share_text(
            (event.get_message_str() or "")
            + " "
            + " ".join(
                _json.dumps(getattr(c, "data", {}) or {}, ensure_ascii=False)
                for c in event.get_messages()
                if isinstance(c, Json)
            )
        )
        logger.info(
            f"[Video] 触发：url={target or '（无链接，走引用视频）'}"
            f" quoted={'有' if quoted else '无'}"
        )
        await self._send(event, "收到视频，正在解析，稍等一会儿～")
        if quoted is not None:
            # 这条消息归视频插件处理，别让 ASR 插件把它当语音转写
            event.stop_event()
        task = asyncio.create_task(self._job(event, target, quoted, self._started_at))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @staticmethod
    def _collect_urls(event: AstrMessageEvent) -> list[str]:
        """纯文本 + JSON 卡片里的链接都要，去重保序。"""
        found: list[str] = list(URL_RE.findall(event.get_message_str() or ""))
        for comp in event.get_messages():
            if isinstance(comp, Json):
                try:
                    blob = _json.dumps(getattr(comp, "data", {}) or {}, ensure_ascii=False)
                except Exception:  # noqa: BLE001
                    continue
                found.extend(URL_RE.findall(blob))
        seen: set[str] = set()
        urls: list[str] = []
        for url in found:
            url = url.rstrip("，。！？、）)】]\"'")
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
        return urls

    @staticmethod
    def _quoted_video(event: AstrMessageEvent):
        """引用消息里的视频（Video 优先）或视频文件（Record）都认。"""
        chains = []
        for comp in event.get_messages():
            if isinstance(comp, Reply) and comp.chain:
                chains.append(comp.chain)
        # 直接发的视频消息也接受
        chains.append(event.get_messages())
        for chain in chains:
            for sub in chain:
                if isinstance(sub, Video) and (
                    getattr(sub, "file", None) or getattr(sub, "path", None)
                ):
                    return sub
            for sub in chain:
                if isinstance(sub, Record) and getattr(sub, "file", None):
                    return sub
        return None

    # ---------------- 任务 ----------------
    async def _send(self, event: AstrMessageEvent, text: str) -> None:
        try:
            await event.send(event.plain_result(text))
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[Video] 发送失败: {exc}")

    async def _job(
        self, event: AstrMessageEvent, url: str | None, quoted, since: float
    ) -> None:
        async with self._lock:
            started = time.time()
            self._busy = True
            self._last_used = started
            try:
                loop = asyncio.get_running_loop()

                # 引用视频优先：直接落盘成文件
                if quoted is not None:
                    try:
                        path = await quoted.convert_to_file_path()
                        video = Path(path)
                        logger.info(f"[Video] 使用引用视频 {video.name}")
                        from .core import media as _media

                        uid = (_media.probe_tags(video).get("comment") or "").strip()
                        if uid and uid in self._uid_meta:
                            self._meta = dict(self._uid_meta[uid])
                            logger.info(
                                f"[Video] 复用该视频指纹的历史标题/标签：{self._meta.get('title')!r}"
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f"[Video] 引用视频取文件失败: {exc}")
                        video = None
                else:
                    video = None

                if video is None and url:
                    mapped = self._url_map.get(url)
                    if mapped and Path(mapped).is_file():
                        video = Path(mapped)
                        logger.info(f"[Video] 同一链接复用上次的文件 {video.name}")
                    else:
                        video = source.find_cached(url)
                if not video and url:
                    video = await loop.run_in_executor(
                        None, lambda: source.wait_cached(url, timeout=5.0)
                    )

                # 解析器常常比我们先跑完：往前看 3 分钟，取最新的未用文件
                if not video:
                    video = source.recent_video(
                        since - 180.0,
                        cache_dir=self._cache_dir(),
                        exclude=self._consumed,
                    )
                    if video:
                        logger.info(f"[Video] 使用最近下载的视频 {video.name}")
                        if url:
                            self._url_map[url] = str(video)
                if not video and url:
                    # 短链被 parser 重定向过，哈希对不上 → 按"新出现的视频"兜底
                    logger.info("[Video] 哈希未命中，改为等待新缓存文件")
                    video = await loop.run_in_executor(
                        None,
                        lambda: source.wait_new_video(
                            since=max(0.0, since - 5.0),
                            timeout=float(self.config["parse_wait"]),
                            cache_dir=self._cache_dir(),
                        ),
                    )
                if not video:
                    recent = sorted(
                        (
                            p
                            for p in source.CACHE_DIR.iterdir()
                            if p.is_file()
                        ),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )[:3] if source.CACHE_DIR.is_dir() else []
                    detail = "，".join(
                        f"{p.name}({p.stat().st_size/1024**2:.1f}MB)" for p in recent
                    ) or "缓存为空"
                    logger.warning(f"[Video] 未匹配到视频；最近缓存：{detail}")
                    await event.send(
                        event.plain_result(
                            "没等到视频文件：可能解析器没处理这个链接，"
                            "或平台需要登录。稍后可以再试一次。"
                        )
                    )
                    return

                workdir = self._workdir() / f"job_{int(started)}_{video.stem[:8]}"
                workdir.mkdir(parents=True, exist_ok=True)
                self._consumed.add(str(video))
                logger.info(f"[Video] 命中缓存 {video.name}（{video.stat().st_size/1024**2:.1f} MB）")

                async def progress(msg: str) -> None:
                    await self._send(event, f"（解析中）{msg}")

                timeout = int(self.config.get("job_timeout_seconds", 900) or 900)
                bundle = await asyncio.wait_for(
                    analyze_video(
                        video,
                        workdir,
                    llama_dir=str(self.config["llama_dir"]),
                    model_path=str(self.config["model_path"]),
                    mmproj_path=str(self.config["mmproj_path"]),
                    asr_model_dir=str(self.config["asr_model_dir"]),
                    port=int(self.config["port"]),
                    ocr_threads=int(self.config["ocr_threads"]) or None,
                    vlm_threads=int(self.config["vlm_threads"]) or None,
                    asr_threads=int(self.config["asr_threads"]) or None,
                    frame_cap=int(self.config["frame_cap"]) or None,
                    vlm_width=int(self.config.get("vlm_image_width", 640) or 640),
                    keep_server=bool(self.config["keep_server"]),
                    term_correction=bool(self.config.get("term_correction", False)),
                        log=lambda msg: logger.info(f"[Video] {msg}"),
                        progress=progress,
                    meta=getattr(self, "_meta", None),
                    ),
                    timeout=timeout,
                )

                meta_used = bundle.get("source_meta") or {}
                uid = meta_used.get("video_uid")
                if uid:
                    self._uid_meta[uid] = meta_used

                text = await self._compose_reply(event, bundle, time.time() - started)
                await event.send(event.plain_result(text))
                self._last_used = time.time()
                self._schedule_idle_unload()
                logger.info(f"[Video] 完成，用时 {time.time()-started:.1f}s")
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[Video] 分析失败: {exc}", exc_info=True)
                try:
                    await event.send(event.plain_result(f"视频分析失败：{str(exc)[:120]}"))
                except Exception:  # noqa: BLE001
                    pass
            finally:
                self._busy = False
                self._last_used = time.time()

    # ---------------- 回复 ----------------
    async def _persona_prompt(self, event: AstrMessageEvent) -> tuple[str, str, str]:
        """返回 (人格提示词, 人设卡名称, 人设卡 id)；日志只打名称。"""
        manager = getattr(self.context, "persona_manager", None)
        if manager is None:
            return "", "", ""

        conversation_persona_id = None
        try:
            cm = getattr(self.context, "conversation_manager", None)
            umo = event.unified_msg_origin
            cid = await cm.get_curr_conversation_id(umo) if cm else None
            if cid:
                conv = await cm.get_conversation(umo, cid)
                conversation_persona_id = getattr(conv, "persona_id", None)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[Video] 读取会话人格 id 失败: {exc}")

        provider_settings = {}
        try:
            provider_settings = (self.context.get_config() or {}).get("provider_settings", {})
        except Exception:  # noqa: BLE001
            provider_settings = {}

        def _fields(persona) -> tuple[str, str, str]:
            if isinstance(persona, dict):
                return (
                    persona.get("prompt", "") or "",
                    persona.get("name", "") or "",
                    persona.get("id", "") or "",
                )
            return (
                getattr(persona, "prompt", "") or "",
                getattr(persona, "name", "") or "",
                getattr(persona, "id", "") or "",
            )

        try:
            persona_id, persona, _forced, _webchat = await manager.resolve_selected_persona(
                umo=event.unified_msg_origin,
                conversation_persona_id=conversation_persona_id,
                platform_name=event.get_platform_name(),
                provider_settings=provider_settings,
            )
            prompt, name, pid = _fields(persona)
            if prompt:
                logger.info(f"[Video] 使用人设卡：{name or pid or persona_id}")
                return prompt, name or str(persona_id or pid), str(persona_id or pid)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[Video] resolve_selected_persona 失败: {exc}")

        try:
            persona = await manager.get_default_persona_v3(umo=event.unified_msg_origin)
            prompt, name, pid = _fields(persona)
            if prompt:
                logger.info(f"[Video] 使用人设卡：{name or pid}（默认）")
            return prompt, name or pid, pid
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[Video] 读取人格失败，使用空人格: {exc}")
            return "", "", ""

    async def _compose_reply(
        self, event: AstrMessageEvent, bundle: dict, elapsed: float
    ) -> str:
        """用会话人格作系统提示，让大模型以自己的口吻转述。"""
        fallback = self._render_reply(bundle, elapsed)
        if not self.config.get("use_chat_model", True):
            return fallback

        digest = bundle.get("digest") or {}
        facts_text = summarize.render_text(digest)

        persona_prompt, persona_name, _persona_id = await self._persona_prompt(event)
        system_prompt = SYSTEM_TEMPLATE.format(persona=persona_prompt or "（无人设）")

        try:
            provider_id = await self.context.get_current_chat_provider_id(
                umo=event.unified_msg_origin
            )
            logger.info(
                "[Video] 发给外部模型：人设卡=%s，系统提示 %d 字，事实材料 %d 字",
                persona_name or "无",
                len(system_prompt),
                len(facts_text),
            )
            logger.debug("[Video] 事实材料预览：%s", facts_text[:600].replace("\n", " | "))
            resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=facts_text,
                system_prompt=system_prompt,
            )
            text = (getattr(resp, "completion_text", "") or "").strip()
            if text:
                bad = summarize.verify_numbers(text, digest.get("facts") or {})
                if bad:
                    logger.warning(f"[Video] 输出含未证实数字 {bad}，回退本地摘要")
                    return fallback
                return text[: int(self.config.get("reply_max_chars", 1200))]
            logger.warning("[Video] 会话模型返回空，回退本地摘要")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[Video] 会话模型成文失败，回退本地摘要: {exc}")
        return fallback

    # ---------------- 后台服务 ----------------
    def _schedule_idle_unload(self) -> None:
        """常驻模式下，闲置超过设定秒数就关掉视觉服务；只保留一个定时器。"""
        idle = int(self.config.get("idle_unload_seconds", 600) or 0)
        if idle <= 0 or not self.config.get("keep_server", True):
            return
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        task = asyncio.create_task(self._idle_unload(idle))
        self._idle_task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _idle_unload(self, idle: int) -> None:
        await asyncio.sleep(idle)
        if self._busy:
            logger.info("[Video] 空闲计时到点但仍有任务在跑，跳过卸载")
            return
        if time.time() - self._last_used < idle:
            return
        port = int(self.config.get("port", 8090))
        try:
            import subprocess

            logger.info(f"[Video] 空闲 {idle}s，关闭视觉服务（端口 {port}）")
            script = (
                "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | "
                f"Where-Object {{ $_.CommandLine -match '--port {port}' }} | "
                "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", script], timeout=20)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[Video] 空闲卸载失败: {exc}")

    # ---------------- 工具 ----------------
    def _enabled(self) -> bool:
        return bool(self.config.get("enable", True))

    def _pick_url(self, urls: list[str]) -> str | None:
        for url in urls:
            if source.find_cached(url) or self._looks_like_video(url):
                return url
        return urls[0] if urls else None

    @staticmethod
    def _looks_like_video(url: str) -> bool:
        low = url.lower()
        return any(host in low for host in VIDEO_HOSTS)

    def _cache_dir(self) -> Path:
        configured = str(self.config.get("parser_cache_dir") or "").strip()
        return Path(configured) if configured else source.CACHE_DIR

    def _workdir(self) -> Path:
        try:
            base = StarTools.get_data_dir("astrbot_plugin_video_understanding")
        except Exception:  # noqa: BLE001
            base = Path("data") / "plugin_data" / "astrbot_plugin_video_understanding"
        path = Path(base) / "jobs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _render_reply(self, bundle: dict, elapsed: float) -> str:
        digest = bundle.get("digest") or {}
        facts = digest.get("facts") or {}
        if not (facts.get("timeline") or facts.get("title")):
            return "视频里没解析到内容（可能没有音轨、画面全黑或文件不完整）。"
        summary = (digest.get("summary") or "").strip()
        timing = bundle.get("timing") or {}
        lines: list[str] = []

        if summary:
            lines.append(summary)
        else:
            lines.append("（摘要生成失败，下面是识别到的事实）")

        if facts.get("title"):
            lines.append("标题：" + "；".join(facts["title"][:4]))
        if facts.get("numbers"):
            lines.append("数字：" + "、".join(facts["numbers"][:10]))
        conflicts = facts.get("conflicts") or []
        if conflicts:
            first = conflicts[0]
            lines.append(f"待核对：{first.get('a')} ↔ {first.get('b')}")
        if digest.get("verified") is False:
            lines.append("注意：摘要里有未在事实中出现的数字：" +
                         "、".join(digest.get("unknown_numbers") or []))

        lines.append(
            f"用时 {elapsed:.0f}s（视频 {timing.get('video_seconds')}s，"
            f"{timing.get('realtime_factor')}x 实时）"
        )
        text = "\n".join(lines)
        limit = int(self.config.get("reply_max_chars", 1200))
        return text[:limit]
