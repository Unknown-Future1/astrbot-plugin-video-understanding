"""两路联调（v2）：场景切点判型 → ASR → OCR ∥ VLM（并行）→ 证据。

用法：
    python run_pipeline.py <video> [--max-frames 12] [--ocr-threads 4] [--server-threads 6]

设计要点：
  · 判型只看 ffmpeg 场景切点密度（边界情况用 3 帧像素差复核），不再跑 8 帧粗筛
  · ASR 先跑（独占 CPU），随后 OCR（限线程）与 VLM（限线程）并行
  · 结尾输出：视频时长、处理总时长、实时比、各阶段耗时
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from astrbot_plugin_video_understanding.core import evidence as ev  # noqa: E402
from astrbot_plugin_video_understanding.core import media, planner, sampling, scene  # noqa: E402
from astrbot_plugin_video_understanding.core import settings, summarize  # noqa: E402
from astrbot_plugin_video_understanding.core.asr import AsrService  # noqa: E402
from astrbot_plugin_video_understanding.core.ocr import OcrService  # noqa: E402
from astrbot_plugin_video_understanding.core.ocr_clean import read_clean  # noqa: E402
from astrbot_plugin_video_understanding.core.vision import VisionService  # noqa: E402

MODEL_DIR = Path(r"H:\Deepseek-Harness\Workspace\model\qwen3.5-4b")
LLAMA = Path(r"H:\Deepseek-Harness\Workspace\model\llama\bin-vulkan")
ASR_DIR = Path(r"H:\Deepseek-Harness\Workspace\asr\models\sensevoice")


async def run(video: Path, args) -> dict:
    job_id = uuid.uuid4().hex[:8]
    work = video.parent / f".pipeline_{video.stem}"
    frames_dir = work / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    stages: dict[str, float] = {}
    t_all = time.time()

    info = media.probe(video)
    print(f"  探测：{info.duration:.1f}s {info.width}x{info.height} "
          f"fps={info.fps:.1f} 音轨={'有' if info.has_audio else '无'}")

    ocr_threads_configured = settings.resolve_threads(args.ocr_threads, kind="ocr")
    # 与 VLM 并行时，OCR 线程压到 4：实测 10 线程会让 VLM 从 4.0s/帧 掉到 6.9s/帧
    ocr_threads = max(1, min(ocr_threads_configured, settings.PARALLEL_OCR_THREADS))
    asr_threads = settings.resolve_threads(args.asr_threads, kind="asr")
    vlm_threads = settings.resolve_threads(args.server_threads, kind="vlm")

    vision = VisionService(
        server_exe=str(LLAMA / "llama-server.exe"),
        model_path=str(MODEL_DIR / "Qwen3.5-4B-Q4_K_M.gguf"),
        mmproj_path=str(MODEL_DIR / "mmproj-F16.gguf"),
        port=args.port,
        ctx_size=4096,
        threads=vlm_threads,
    )
    boot: dict = {"done": False, "started": False, "seconds": 0.0, "error": ""}

    def boot_server() -> None:
        t_start = time.time()
        try:
            boot["started"] = vision.ensure_server()
            boot["done"] = True
        except Exception as exc:  # noqa: BLE001
            boot["error"] = str(exc)[:200]
        boot["seconds"] = round(time.time() - t_start, 2)

    boot_thread = threading.Thread(target=boot_server, daemon=True)
    boot_thread.start()

    # ---------- ① 判型：只依赖场景切点 ----------
    t0 = time.time()
    cuts = scene.scene_cuts(video)
    stages["scene"] = round(time.time() - t0, 2)
    stats = scene.scene_stats(cuts, info.duration)
    profile = "dynamic" if stats["per_minute"] >= 8.0 else "talk"

    # 边界（3~8 次/分钟）用 3 帧像素差复核
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
    print(f"  判型：{profile}（切点 {stats['cuts']} 个 = {stats['per_minute']}/min"
          + (f"，复核像素 {pixel_mean:.3f}" if pixel_mean else "") + f"）| {stages['scene']:.1f}s")

    # ---------- ② 计划 ----------
    budget = args.max_frames or settings.frame_budget(info.duration)
    if profile == "talk":
        vlm_times = sampling.coarse_times(info.duration, min(4, budget))
        ocr_times = sampling.ocr_times_talk(info.duration, interval=5.0)
    else:
        vlm_times = planner.plan_timeline(info.duration, cuts, cap=budget)
        ocr_times = list(vlm_times)
    print(f"  计划：VLM {len(vlm_times)} 帧 | OCR {len(ocr_times)} 帧")

    asr = AsrService(model_dir=str(ASR_DIR), num_threads=asr_threads)
    ocr = OcrService(num_threads=ocr_threads)
    ocr_ok, ocr_err = ocr.available()
    print(f"  OCR：{'可用' if ocr_ok else '不可用（' + ocr_err[:40] + '）'}"
          f"，并行线程 {ocr_threads}（配置 {ocr_threads_configured}）"
          f" | VLM {vlm_threads} 线程 / ASR {asr_threads} 线程")

    # 抽帧（OCR 与 VLM 共用）
    t0 = time.time()
    extracted: dict[float, str] = {}
    for ts in sorted(set(ocr_times) | set(vlm_times)):
        name = f"{ts:07.3f}.jpg"
        path = frames_dir / name
        if not path.is_file():
            media.frame_at(video, ts, path, width=896 if ts in vlm_times else 720)
        if path.is_file():
            extracted[ts] = name
    stages["extract"] = round(time.time() - t0, 2)
    print(f"  抽帧：{len(extracted)} 张 / {stages['extract']:.1f}s")

    # ---------- ③ ASR（先独占 CPU）----------
    speech: dict = {"status": "not_present", "entries": []}
    if info.has_audio:
        t0 = time.time()
        wav = media.extract_audio(video, work / "audio.wav")
        loop = asyncio.get_running_loop()
        speech = await loop.run_in_executor(None, asr.transcribe, wav)
        stages["asr"] = round(time.time() - t0, 2)
        print(f"  ASR：{speech.get('status')} {len(speech.get('entries') or [])} 句 / {stages['asr']}s")
    else:
        stages["asr"] = 0.0
        print("  ASR：无音轨，跳过")

    # ---------- ④ OCR ∥ VLM ----------
    ocr_targets = [ts for ts in sorted(extracted) if ocr_ok and ts in set(ocr_times)]
    ocr_results: dict[str, dict] = {}
    vlm_targets = [ts for ts in sorted(vlm_times) if ts in extracted]

    t_par = time.time()
    loop = asyncio.get_running_loop()

    def do_ocr() -> tuple[dict[str, dict], float]:
        started = time.time()
        out: dict[str, dict] = {}
        for ts in ocr_targets:
            name = extracted[ts]
            out[name] = read_clean(ocr, frames_dir / name)
        return out, round(time.time() - started, 2)

    ocr_task = loop.run_in_executor(None, do_ocr) if ocr_targets else None

    boot_thread.join(timeout=180)
    if not boot["done"] or boot["error"]:
        raise RuntimeError(f"视觉服务启动失败：{boot['error'] or '超时'}")
    stages["server_boot"] = boot["seconds"]
    print(f"  视觉服务：{'新启动' if boot['started'] else '复用'}（{boot['seconds']:.1f}s，"
          f"已与场景扫描重叠）")
    t_infer = time.time()
    described: dict[float, dict] = {}
    for ts in vlm_targets:
        result = vision.describe(frames_dir / extracted[ts], ts)
        described[ts] = result
        print(f"  [{ts:6.2f}s] VLM {result['seconds']:4.1f}s  {result.get('scene','')[:40]}")
    stages["vlm_infer"] = round(time.time() - t_infer, 2)

    if ocr_task is not None:
        ocr_results, ocr_task_seconds = await ocr_task
        stages["ocr_task"] = ocr_task_seconds
    stages["ocr_and_vlm"] = round(time.time() - t_par, 2)
    print(f"  OCR ∥ VLM 并行：{stages['ocr_and_vlm']:.1f}s"
          f"（OCR {len(ocr_results)} 帧 用时 {stages.get('ocr_task', 0)}s / "
          f"VLM {len(described)} 帧 用时 {stages['vlm_infer']}s）")

    # ---------- ⑤ talk 类按 OCR 文本变化补 VLM ----------
    if profile == "talk" and ocr_results and len(described) < 5:
        changed, prev = [], None
        for ts in sorted(ocr_results):
            body = " | ".join(line["text"] for line in ocr_results[ts].get("body") or [])
            if prev is not None and body and body != prev:
                changed.append(ts)
            prev = body or prev
        extra = [ts for ts in changed[:2] if ts not in described]
        for ts in extra:
            ts_float = float(Path(ts).stem)
            result = vision.describe(frames_dir / ts, ts_float)
            described[ts_float] = result
            print(f"  [补帧 {ts_float:6.2f}s] {result.get('scene','')[:36]}")

    # ---------- ⑥ 组装 ----------
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
        frame["scene"] = vlm_points[nearest].get("scene", "")
        frame["derived_scene"] = round(nearest, 3)

    total = time.time() - t_all
    stages["total"] = round(total, 2)

    bundle = ev.build(
        job_id=job_id,
        info=info.as_dict(),
        frames=frames,
        speech=speech,
        workdir=work,
        visual_model="qwen3.5-4b + rapidocr",
    )
    body_lines = sum(1 for info_ in ocr_results.values() for _ in info_.get("body") or [])
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
        "seconds": round(stages.get("ocr", 0.0), 1),
    }
    bundle["sampling"] = {
        "profile": profile,
        "scene_cuts": stats["cuts"],
        "cuts_per_minute": stats["per_minute"],
        "vlm_frames": len(described),
        "ocr_frames": len(ocr_results),
        "pixel_check": round(pixel_mean, 3),
    }
    bundle["timing"] = {
        "video_seconds": round(info.duration, 2),
        "total_seconds": round(total, 2),
        "realtime_factor": round(info.duration / total, 2) if total else 0.0,
        "stages": stages,
    }

    # ---------- ⑦ 成文（本地模型写摘要，事实来自规则层）----------
    t0 = time.time()
    digest = summarize.build_digest(bundle, service=vision)
    stages["summarize"] = round(time.time() - t0, 2)
    bundle["digest"] = digest
    print(f"  成文：{stages['summarize']}s，摘要 {len(digest.get('summary') or '')} 字")
    if digest.get("summary"):
        print("  ── 摘要 ──")
        for line in digest["summary"].splitlines()[:6]:
            print(f"  {line}")

    out = work / "result.json"
    out.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  视频时长 {info.duration:.1f}s | 处理总时长 {total:.1f}s | 实时比 "
          f"{info.duration / total:.2f}x")
    print(f"  阶段：{json.dumps(stages, ensure_ascii=False)}")
    print(f"  证据 → {out}")

    if boot['started'] and not args.keep_server:
        vision.shutdown()
        print("  已关闭视觉服务")
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="视觉帧上限，默认按时长自适应（6~15）")
    ap.add_argument("--ocr-threads", type=int, default=None,
                    help="OCR 线程数，默认一半核心（封顶 10）")
    ap.add_argument("--asr-threads", type=int, default=None,
                    help="ASR 线程数，默认四分之一核心（封顶 8）")
    ap.add_argument("--server-threads", type=int, default=None,
                    help="视觉服务线程数，默认三分之一核心（封顶 8）")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--keep-server", action="store_true")
    args = ap.parse_args()
    video = Path(args.video)
    if not video.is_file():
        raise SystemExit(f"找不到视频：{video}")
    asyncio.run(run(video, args))


if __name__ == "__main__":
    main()
