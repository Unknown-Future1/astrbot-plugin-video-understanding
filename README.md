# ⚠️ 半成品（WIP）

> 基础功能已实现，仍在修改完善；接口、目录结构和参数可能随时变动，请勿用于生产。

---

# 视频理解（AstrBot 插件）

`@机器人 + 视频链接`，自动抽帧、OCR、语音识别（ASR），最后汇总成文字。

## 功能

| 能力 | 说明 |
| --- | --- |
| 视频抽帧 | 按场景/时间采样，控制帧数上限 |
| OCR | 画面文字识别，清理噪声 |
| ASR | SenseVoice 本地语音识别，输出字幕/要点 |
| 汇总 | 结合画面与语音生成文字总结 |
| 本地模型 | 视觉模型走 llama.cpp（GGUF + mmproj） |
| 空闲卸载 | 可配置闲置后卸载视觉服务，释放显存 |

## 使用

```
@机器人  https://www.bilibili.com/video/BVxxxx
```

或引用视频链接后 @ 机器人。

## 安装

1. 把本目录放到 `AstrBot/data/plugins/`
2. 安装 `requirements.txt`
3. 在插件配置里填：
   - 视觉模型 GGUF 路径
   - mmproj 路径
   - SenseVoice 模型目录
4. 重启 AstrBot

## 配置项

见 `_conf_schema.json`，主要包括：

- `enable` 总开关
- `llama_dir` llama.cpp 可执行目录
- `model_path` / `mmproj_path` 视觉模型
- `asr_model_dir` SenseVoice 模型目录
- `port` 视觉服务端口
- `frame_cap` 抽帧上限
- `keep_server` 是否常驻
- `idle_unload_seconds` 空闲卸载时间

## 状态

- 版本：0.1.0
- 状态：**半成品**，基础链路可跑，细节仍在完善
- 问题/建议：欢迎提 Issue

## License

未定（暂未添加）
