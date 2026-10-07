# 从独立 STT/TTS 服务迁移

迁移工具将模型、参考音色、CosyVoice/Matcha 源码及 Python 运行库复制到统一服务目录。原服务目录保留，供验证与回退。

```text
python migrate_assets.py --root <new-service> --tts <old-tts-service> --stt <old-stt-service>
python migrate_assets.py --root <new-service> --tts <old-tts-service> --stt <old-stt-service> --apply
```

工具执行逐文件 SHA-256 校验，调整环境路径和启动器，并在 `runtime/asset-migration-*` 保存本机迁移记录。迁移前备份原配置；完成后核对 `service.local.json` 中的模型、源码、解释器和 ffmpeg 路径。

## 目标布局

- `models/stt`：STT 模型。
- `models/tts`：TTS 模型及参考音色。
- `cosyvoice-src`：第三方源码。
- `venv`：独立 Python 环境。
- `venv/Lib/speech-stt-site-packages`：Windows 迁移时补充的 STT 依赖。

迁移环境仍可能依赖系统基础 Python、ffmpeg 及用户级缓存。检查这些公共位置，确认它们独立于待删除的旧目录。

## 验收

1. 运行服务契约测试。
2. 使用 `probe_local.py` 完成真实模型预热、合成、转写和就绪检查。
3. 核对启动任务、系统服务、插件配置及解释器搜索路径。
4. 完成客户端录音、播放、打断与下一轮恢复测试。
5. 保留回退副本，验收后再清理旧目录。

可将旧目录的 JSON 数组写入环境变量 `SPEECH_VALIDATION_BLOCKED_ROOTS`，再运行探针。验证进程会阻止读取这些目录并记录加载的库路径，用于确认新服务的独立性。

迁移报告、日志和本机配置留在本地；发布仓库时排除其中的绝对路径、用户信息、音频和转写内容。
