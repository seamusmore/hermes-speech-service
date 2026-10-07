# Hermes Speech Service

为 Hermes 及其他客户端提供本地 STT/TTS HTTP API。一个进程、一个端口，分别管理转写与合成引擎、请求队列和取消事件。

配套客户端：[hermes-speech](https://github.com/seamusmore/hermes-speech)。服务可独立部署，使用自己的模型和 Python 环境。

## 快速启动

```bash
git clone https://github.com/seamusmore/hermes-speech-service.git
cd hermes-speech-service
python -m venv .venv
source .venv/bin/activate
python -m pip install ".[stt]"
```

Windows 使用 `.venv\Scripts\Activate.ps1` 激活环境。Python 版本应与所选模型运行库兼容；现有 Windows 运行环境使用 Python 3.11。推理环境需要 ffmpeg。

上述命令安装服务与 STT 依赖。TTS 还需要单独准备 CosyVoice 源码、模型、参考音色以及与模型兼容的 PyTorch、Transformers、Matcha-TTS 等依赖。模型与第三方源码各自遵守其许可证。

复制 `service.example.json` 为本机 `service.local.json`，填写实际模型与源码路径，再启动：

```bash
python run.py --config service.local.json --host 127.0.0.1 --port 8000
```

示例中的相对路径以启动工作目录为基准；生产部署建议填写本机绝对路径。`service.local.json`、环境变量、模型、venv 和运行记录由 Git 忽略。

## API

| 端点 | 说明 |
|---|---|
| `GET /health` | 进程及协议检查，保持模型按需加载 |
| `GET /capabilities` | 服务能力与协议 |
| `GET /ready` | STT/TTS 预热完成时返回 200，否则 503 |
| `POST /warmup` | 加载并预热两个模型 |
| `POST /stt/transcribe` | multipart 音频 `file`，可选 `language` |
| `POST /tts/synthesize` | multipart `text`，可选 `voice`、`model`，返回音频 |
| `POST /tts/synthesize-stream` | SSE 音频块，PCM16/base64，24 kHz 单声道 |
| `POST /tts/cancel/{request_id}` | 取消指定合成请求 |

组件接口还提供 `/stt` 与 `/tts` 下的健康、模型及缓存管理能力。

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/warmup
curl -F "file=@sample.wav" http://127.0.0.1:8000/stt/transcribe
curl -F "text=语音服务测试。" http://127.0.0.1:8000/tts/synthesize -o output.wav
```

## 引擎

| 能力 | 引擎 | 运行库 |
|---|---|---|
| STT | `sensevoice-q8` | sherpa-onnx / INT8 |
| STT | `sensevoice` | FunASR / PyTorch |
| STT | `whisper` | faster-whisper |
| TTS | `cosyvoice2`、`cosyvoice3` | CosyVoice 及其配套依赖 |

模型目录与默认音色由部署配置提供。各引擎保留自己的加载、下载和空闲卸载行为。

## 配置

| 环境变量 | 用途 |
|---|---|
| `STT_ENGINE`、`TTS_ENGINE` | 选择引擎 |
| `STT_MODEL_DIR`、`TTS_MODEL_DIR` | 本机模型目录 |
| `COSYVOICE_SRC_DIR` | CosyVoice 源码目录 |
| `TTS_DEFAULT_VOICE` | 本机默认参考音色文件 |
| `SPEECH_PRELOAD` | 设置为 `1` 时启动预热 |
| `SPEECH_FFMPEG_DIR` | 可选 ffmpeg 程序目录 |
| `HERMES_SPEECH_SERVICE_TOKEN` | 服务 Bearer token，通过环境设置 |

环境变量优先于 JSON 配置。JSON 的 `environment` 接受 `STT_`、`TTS_`、`SPEECH_`、`COSYVOICE_` 前缀的部署设置。

## 远程部署

监听非回环地址时必须设置 `HERMES_SPEECH_SERVICE_TOKEN`。启用认证后，模型、缓存、预热及就绪请求使用 `Authorization: Bearer <token>`；`/health` 提供基础存活信息。通过 HTTPS 反向代理供远程客户端访问，单进程启动以保持模型和取消状态一致。

## 代码结构

- `src/hermes_speech_service/app.py`：统一应用与认证。
- `src/hermes_speech_service/stt`：转写引擎及接口。
- `src/hermes_speech_service/tts`：合成引擎、流式输出及取消。
- `run.py`、`start-service.ps1`：启动入口。
- `tests`：HTTP 契约与生命周期测试。
- `MIGRATION.md`：从独立 STT/TTS 服务迁移的通用步骤。

## 验证与运维

```bash
python -m unittest discover -s tests
python probe_local.py
```

契约测试使用隔离的模型替身。`probe_local.py` 使用实际模型启动临时服务，执行预热、合成、转写和就绪检查，结果写入本机 `runtime`。该测试需要完整模型依赖，不测量扬声器和麦克风体验。

将日志作为本机诊断资料，分享前清除凭据、转写内容、会话标识和个人路径。服务通过 `/ready` 报告模型就绪状态；冷启动时间取决于模型、设备及缓存。

## 许可证

[MIT](LICENSE)。第三方模型与运行库遵守各自的许可证。
