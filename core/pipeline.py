"""可复用的分析主流程（插件与命令行共用）。

与 proto/run_pipeline.py 等价，区别只有：
  · 不打印，改成 log 回调
  · 模型路径/线程/帧预算可传入
  · 返回 bundle（含 digest）
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from pathlib import Path

from . import evidence as ev
from . import terms
from . import media, planner, sampling, scene, settings, summarize
from .asr import AsrService
from .ocr import OcrService
from .ocr_clean import read_clean
from .vision import VisionService


async def analyze_video(
    video: Path,
    workdir: Path,
    *,
    llama_dir: str,
    model_path: str,
    mmproj_path: str,
    asr_model_dir: str,
    port: int = 8090,
    ocr_threads: int | None = None,
    vlm_threads: int | None = None,
    asr_threads: int | None = None,
    frame_cap: int | None = None,
    vlm_width: int = 640,
    progress=None,
    meta: dict | None = None,
    keep_server: bool = True,
    term_correction: bool = False,
    log=None,
) -> dict:
    say = log or (lambda _msg: None)
    notify = progress or (lambda _msg: None)
    job_id = uuid.uuid4().hex[:8]
    frames_dir = workdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    stages: dict[str, float] = {}
    t_all = time.time()

    info = media.probe(video)
    say(f"探测 {info.duration:.1f}s {info.width}x{info.height} 音轨={'有' if info.has_audio else '无'}")
    tags = media.probe_tags(video)
    uid = (tags.get("comment") or "").strip()
    if meta is None:
        meta = {}
    if not (meta.get("title") or meta.get("tags")) and (tags.get("title") or tags.get("description")):
        meta = {
            "title": (tags.get("title") or tags.get("description") or "")[:80],
            "tags": [],
            "from": "container",
        }
    meta["video_uid"] = uid
    if meta.get("title") or meta.get("tags"):
        say(f"分享文案：标题={meta.get('title')!r} 标签={meta.get('tags')}")
    elif uid:
        say(f"分享文案：无（视频指纹 {uid}，可用于复用历史标题/标签）")

    ocr_threads_cfg = settings.resolve_threads(ocr_threads, kind="ocr")
    ocr_workers = max(1, min(ocr_threads_cfg, settings.PARALLEL_OCR_THREADS))
    vlm_workers = settings.resolve_threads(vlm_threads, kind="vlm")
    asr_workers = settings.resolve_threads(asr_threads, kind="asr")

    vision = VisionService(
        server_exe=str(Path(llama_dir) / "llama-server.exe"),
        model_path=model_path,
        mmproj_path=mmproj_path,
        port=port,
        ctx_size=4096,
        threads=vlm_workers,
    )
    boot = {"done": False, "started": False, "seconds": 0.0, "error": ""}

    def boot_server() -> None:
        t0 = time.time()
        try:
            boot["started"] = vision.ensure_server()
            boot["done"] = True
        except Exception as exc:  # noqa: BLE001
            boot["error"] = str(exc)[:200]
        boot["seconds"] = round(time.time() - t0, 2)

    boot_thread = threading.Thread(target=boot_server, daemon=True)
    boot_thread.start()

    # ① 判型
    t0 = time.time()
    cuts = scene.scene_cuts(video)
    stages["scene"] = round(time.time() - t0, 2)
    stats = scene.scene_stats(cuts, info.duration)
    profile = "dynamic" if stats["per_minute"] >= 8.0 else "talk"
    pixel_mean = 0.0
    if 3.0 <= stats["per_minute"] < 8.0:
        probe_times = sampling.coarse_times(info.duration, 3)
        files = []
        for ts in probe_times:
            path = frames_dir / f"{ts:07.3f}.jpg"
            if not path.is_file():
                media.frame_at(video, ts, path, width=640)
            if path.is_file():
                files.append(path)
        diffs = [sampling.pixel_diff(a, b) for a, b in zip(files, files[1:])]
        pixel_mean = sum(diffs) / len(diffs) if diffs else 0.0
        if pixel_mean >= 0.12:
            profile = "dynamic"
    say(f"判型 {profile}（切点 {stats['cuts']} = {stats['per_minute']}/min，"
        f"像素复核 {pixel_mean}）")

    # ② 计划
    budget = frame_cap or settings.frame_budget(info.duration)
    # OCR 间隔随时长放宽：短视频 1 秒，长视频 2~3 秒，避免识别比视频本身还久
    ocr_interval = 1.0 if info.duration <= 60 else (2.0 if info.duration <= 180 else 3.0)
    if profile == "talk":
        vlm_times = sampling.coarse_times(info.duration, min(4, budget))
        ocr_times = sampling.ocr_times_talk(info.duration, interval=ocr_interval, cap=120)
    else:
        vlm_times = planner.plan_timeline(info.duration, cuts, cap=budget)
        ocr_times = sampling.ocr_times_talk(info.duration, interval=ocr_interval, cap=120)

    say(f"计划：VLM {len(vlm_times)} 帧 {vlm_times}")
    say(f"计划：OCR {len(ocr_times)} 帧 {ocr_times}")

    asr = AsrService(model_dir=asr_model_dir, num_threads=asr_workers)
    ocr = OcrService(num_threads=ocr_workers)
    ocr_ok, _err = ocr.available()

    # 无文字视频：先用 2 张粗筛帧预判
    if ocr_ok and profile == "talk":
        probe = sampling.coarse_times(info.duration, 2)
        has_text = False
        for ts in probe:
            path = frames_dir / f"{ts:07.3f}.jpg"
            if not path.is_file():
                media.frame_at(video, ts, path, width=640)
            if path.is_file() and read_clean(ocr, path).get("body"):
                has_text = True
                break
        if not has_text:
            ocr_times = []

    notify(f"抽帧中（{info.duration:.0f}s 视频，OCR {len(ocr_times)} 帧 / 视觉 {len(vlm_times)} 帧）")

    # ③ 抽帧
    t0 = time.time()
    extracted: dict[float, str] = {}
    for ts in sorted(set(ocr_times) | set(vlm_times)):
        name = f"{ts:07.3f}.jpg"
        path = frames_dir / name
        if not path.is_file():
            media.frame_at(video, ts, path, width=vlm_width if ts in vlm_times else 720)
        if path.is_file():
            extracted[ts] = name
    stages["extract"] = round(time.time() - t0, 2)

    notify("抽帧完成，开始语音识别")

    # ④ ASR
    speech: dict = {"status": "not_present", "entries": []}
    if info.has_audio:
        t0 = time.time()
        wav = media.extract_audio(video, workdir / "audio.wav")
        loop = asyncio.get_running_loop()
        speech = await loop.run_in_executor(None, asr.transcribe, wav)
        stages["asr"] = round(time.time() - t0, 2)
        lines = [
            f"ASR {speech.get('status')} {len(speech.get('entries') or [])} 句"
            f"（事件={speech.get('events')} 情绪={speech.get('emotion')}）"
        ]
        for item in speech.get("entries") or []:
            flag = " [疑似不可靠]" if item.get("suspect") else ""
            lines.append(
                f"  {item['t0']:6.2f}-{item['t1']:6.2f}s q={item.get('quality')}"
                f" {item['text']}{flag}"
            )
        say("\n".join(lines))

    notify("语音识别完成，开始画面与文字识别")

    # ⑤ OCR ∥ VLM
    loop = asyncio.get_running_loop()
    ocr_targets = [ts for ts in sorted(extracted) if ocr_ok and ts in set(ocr_times)]
    vlm_targets = [ts for ts in sorted(vlm_times) if ts in extracted]

    def do_ocr():
        started = time.time()
        out = {}
        prev_hash = None
        prev_sig = None
        prev_result = None
        skipped = 0
        for ts in ocr_targets:
            name = extracted[ts]
            path = frames_dir / name
            # 像素几乎没变 → 直接复用上一帧结果，不重跑 OCR
            if prev_hash is not None:
                try:
                    diff = sampling.distance(prev_hash, sampling.dhash(path))
                    if diff < 0.02:
                        out[name] = prev_result
                        skipped += 1
                        continue
                except Exception:  # noqa: BLE001
                    pass
            result = read_clean(ocr, path)
            sig = result.get("text") or ""
            if prev_sig is not None and sig == prev_sig and result.get("body"):
                out[name] = prev_result
                skipped += 1
                continue
            out[name] = result
            prev_result, prev_sig = result, sig
            try:
                prev_hash = sampling.dhash(path)
            except Exception:  # noqa: BLE001
                prev_hash = None
        say(f"OCR 去重跳过 {skipped} 帧（{len(ocr_targets)} → {len(ocr_targets)-skipped} 次识别）")
        return out, round(time.time() - started, 2)

    ocr_task = loop.run_in_executor(None, do_ocr) if ocr_targets else None
    boot_thread.join(timeout=180)
    if not boot["done"] or boot["error"]:
        raise RuntimeError(f"视觉服务启动失败：{boot['error'] or '超时'}")

    t_par = time.time()
    described: dict[float, dict] = {}
    for ts in vlm_targets:
        described[ts] = vision.describe(frames_dir / extracted[ts], ts)
    stages["vlm"] = round(time.time() - t_par, 2)

    ocr_results: dict[str, dict] = {}
    if ocr_task is not None:
        ocr_results, ocr_seconds = await ocr_task
        stages["ocr"] = ocr_seconds

    # ⑥ talk 类补帧（文字变化点）
    if profile == "talk" and ocr_results and len(described) < 5:
        changed, prev = [], None
        for name in sorted(ocr_results):
            body = " | ".join(line["text"] for line in ocr_results[name].get("body") or [])
            if prev is not None and body and body != prev:
                changed.append(name)
            prev = body or prev
        for name in changed[:2]:
            ts = float(Path(name).stem)
            if ts in described:
                continue
            described[ts] = vision.describe(frames_dir / name, ts)

    vlm_lines = [f"视觉：{len(described)} 帧描述"]
    for ts, item in sorted(described.items()):
        vlm_lines.append(f"  {ts:6.2f}s {item.get('scene','')}")
    say("\n".join(vlm_lines))

    ocr_lines = [f"OCR：{len(ocr_results)} 帧"]
    for name, ocr_info in sorted(ocr_results.items()):
        body = " | ".join(line["text"] for line in ocr_info.get("body") or [])
        marks = " | ".join(line["text"] for line in ocr_info.get("watermarks") or [])
        ocr_lines.append(f"  {name} 正文=[{body}] 水印=[{marks}]")
    say("\n".join(ocr_lines))

    # ⑥.5 OCR 与 ASR 对比：找异常点（视觉并行期间 OCR 已跑完，这里只做比对与补帧）
    anomalies: list[dict] = []
    try:
        speech_entries = speech.get("entries") or []
        all_screen = [
            (float(Path(name).stem), line["text"])
            for name, info in (ocr_results or {}).items()
            for line in (info.get("body") or [])
        ]
        for item in speech_entries:
            if item.get("suspect"):
                continue
            near = [
                text for ts, text in all_screen if abs(ts - item["t0"]) <= 4.0
            ]
            if not near and all_screen:
                anomalies.append(
                    {"type": "no_text_nearby", "t": item["t0"], "text": item["text"][:40]}
                )
    except Exception as exc:  # noqa: BLE001
        say(f"异常点比对失败（不影响主流程）: {exc}")

    # 精抽：对异常点补 2~3 帧视觉（不重复已看过的）
    extra_targets = [
        a["t"] for a in anomalies
        if all(abs(a["t"] - t) >= 3.0 for t in described)
    ][:3]
    if extra_targets:
        say(f"异常点精抽 {len(extra_targets)} 帧: {[round(t,2) for t in extra_targets]}")
        for ts in extra_targets:
            name = f"{ts:07.3f}.jpg"
            path = frames_dir / name
            if not path.is_file():
                media.frame_at(video, ts, path, width=vlm_width)
            if path.is_file():
                extracted[ts] = name
                described[ts] = vision.describe(path, ts)
                line = f"  {ts:6.2f}s {described[ts].get('scene','')}"
                say("精抽 " + line.strip())

    # ⑦ 组证据
    frames: list[dict] = []
    for ts in sorted(extracted):
        name = extracted[ts]
        ocr_info = ocr_results.get(name) or {}
        vlm_info = described.get(ts) or {}
        frames.append(
            {
                "t": ts,
                "file": name,
                "ok": True,
                "w": info.width,
                "h": info.height,
                "scene": vlm_info.get("scene", ""),
                "subjects": vlm_info.get("subjects") or [],
                "action": vlm_info.get("action", ""),
                "mood": vlm_info.get("mood", ""),
                "on_screen_text": [line["text"] for line in ocr_info.get("body") or []],
                "watermarks": [line["text"] for line in ocr_info.get("watermarks") or []],
                "uncertain": vlm_info.get("uncertain") or [],
            }
        )

    vlm_points = {t: item for t, item in described.items()}
    for frame in frames:
        if frame["scene"] or not vlm_points:
            continue
        nearest = min(vlm_points, key=lambda t: abs(t - frame["t"]))
        source = vlm_points[nearest]
        frame["scene"] = source.get("scene", "")
        frame["action"] = source.get("action", "")
        frame["mood"] = source.get("mood", "")
        frame["derived_scene"] = round(nearest, 3)

    bundle = ev.build(
        job_id=job_id,
        info=info.as_dict(),
        frames=frames,
        speech=speech,
        workdir=workdir,
        visual_model="qwen3.5-4b + rapidocr",
    )
    body_lines = sum(1 for i in ocr_results.values() for _ in i.get("body") or [])
    if not ocr_ok:
        ocr_status = "disabled"
    elif not ocr_results:
        ocr_status = "failed"
    elif body_lines == 0:
        ocr_status = "no_text"
    else:
        ocr_status = "ok"
    bundle["channels"]["ocr"] = {
        "status": ocr_status,
        "frames": len(ocr_results),
        "text_lines": body_lines,
    }
    bundle["anomalies"] = anomalies
    bundle["source_meta"] = meta or {}
    if ocr_status == "no_text":
        say("OCR 未读到任何文字（无字幕视频，文字路失效）")
    bundle["sampling"] = {
        "profile": profile,
        "scene_cuts": stats["cuts"],
        "cuts_per_minute": stats["per_minute"],
        "vlm_frames": len(described),
        "ocr_frames": len(ocr_results),
        "pixel_check": round(pixel_mean, 3),
    }

    # ⑦.5 术语纠错（默认关闭：交给外部大模型按上下文判断）
    corrections: list[dict] = []
    if term_correction:
        screen_lines = [
            text for frame in frames for text in (frame.get("on_screen_text") or [])
        ]
        texts = [item["text"] for item in bundle["evidence"]["speech_text"]]
        fixed, corrections = terms.correct_speech(texts, screen_lines)
        for item, new_text in zip(bundle["evidence"]["speech_text"], fixed):
            if new_text != item["text"]:
                item["original_text"] = item["text"]
                item["text"] = new_text
                item["corrected"] = True
        if corrections:
            say("术语纠正: " + "；".join(f"{c['from']}→{c['to']}" for c in corrections[:5]))
    bundle["corrections"] = corrections

    notify("画面识别完成，正在整理内容")

    # ⑧ 成文
    t0 = time.time()
    digest = summarize.build_digest(bundle, service=vision)
    stages["summarize"] = round(time.time() - t0, 2)
    bundle["digest"] = digest
    facts = digest.get("facts") or {}
    say(f"标题={facts.get('title')} 数字={facts.get('numbers')}")
    say(f"冲突 {len(facts.get('conflicts') or [])} 条: {facts.get('conflicts')}")
    say(f"摘要：{digest.get('summary')}")

    total = time.time() - t_all
    stages["total"] = round(total, 2)
    bundle["timing"] = {
        "video_seconds": round(info.duration, 2),
        "total_seconds": round(total, 2),
        "realtime_factor": round(info.duration / total, 2) if total else 0.0,
        "stages": stages,
    }
    (workdir / "result.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    say(f"完成 {total:.1f}s（{bundle['timing']['realtime_factor']}x 实时）")

    if not keep_server:
        vision.shutdown()
    return bundle
