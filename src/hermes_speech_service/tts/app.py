#!/usr/bin/env python3
"""Local TTS component mounted by Hermes Speech Service.

Launch the unified service with: python run.py --config service.local.json
Model and source locations are configured by the service environment.
"""

import os
import sys
import time
import threading
import logging
import tempfile
from pathlib import Path
import re
from typing import Optional
from contextlib import asynccontextmanager, closing
from .runtime import ModelRuntime
from .cpu_policy import configure_cpu_policy
from .stream_control import stream_events
from uuid import uuid4
from starlette.concurrency import run_in_threadpool

# 确保项目根目录在 sys.path，支持在任何目录下启动服务
_svc_root = Path(__file__).resolve().parent

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

import base64
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tts-service")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

SERVICE_PORT = int(os.getenv("TTS_SERVICE_PORT", "8002"))
SERVICE_HOST = os.getenv("TTS_SERVICE_HOST", "127.0.0.1")
WORKERS = int(os.getenv("TTS_SERVICE_WORKERS", "1"))

# 引擎配置
_deployment_path = _svc_root / 'deployment.json'
_deployment = json.loads(_deployment_path.read_text(encoding='utf-8')) if _deployment_path.exists() else {}
TTS_ENGINE = os.getenv("TTS_ENGINE", _deployment.get('engine', 'cosyvoice2'))
TTS_MODEL = os.getenv("TTS_MODEL", "")  # 空=引擎默认模型
TTS_DEFAULT_VOICE = os.getenv("TTS_DEFAULT_VOICE", "")
TTS_DEFAULT_LANGUAGE = os.getenv("TTS_DEFAULT_LANGUAGE", "zh")
TTS_OUTPUT_FORMAT = os.getenv("TTS_OUTPUT_FORMAT", "wav")
MAX_TEXT_LENGTH = int(os.getenv("TTS_MAX_TEXT_LENGTH", "5000"))

# ---------------------------------------------------------------------------
# 文本预处理：品牌名 + 单位符号归一化
# ---------------------------------------------------------------------------
_BRAND_MAP = {
    "360": "三六零",
}

_UNIT_PATTERNS = [
    (re.compile(r'(\d+)\s*mAh(?![a-zA-Z])'), r'\1毫安时'),
    (re.compile(r'(\d+)\s*Ah(?![a-zA-Z])'), r'\1安时'),
    (re.compile(r'(\d+)\s*Wh(?![a-zA-Z])'), r'\1瓦时'),
    # 电气（顺序重要：长单位放前面，避免短单位先匹配）
    (re.compile(r'(\d+)\s*mA(?![a-zA-Z])'), r'\1毫安'),
    (re.compile(r'(\d+)\s*MHz(?![a-zA-Z])'), r'\1兆赫兹'),
    (re.compile(r'(\d+)\s*GHz(?![a-zA-Z])'), r'\1吉赫兹'),
    (re.compile(r'(\d+)\s*kHz(?![a-zA-Z])'), r'\1千赫兹'),
    (re.compile(r'(\d+)\s*kW(?![a-zA-Z])'), r'\1千瓦'),
    (re.compile(r'(\d+)\s*Hz(?![a-zA-Z])'), r'\1赫兹'),
    (re.compile(r'(\d+)\s*V(?![a-zA-Z])'), r'\1伏'),
    (re.compile(r'(\d+)\s*A(?![a-zA-Z])'), r'\1安'),
    (re.compile(r'(\d+)\s*W(?![a-zA-Z])'), r'\1瓦'),
    (re.compile(r'(\d+)\s*Ω(?![a-zA-Z])'), r'\1欧姆'),
    # 容量
    (re.compile(r'(\d+)\s*GB(?![a-zA-Z])'), r'\1吉字节'),
    (re.compile(r'(\d+)\s*MB(?![a-zA-Z])'), r'\1兆字节'),
    (re.compile(r'(\d+)\s*KB(?![a-zA-Z])'), r'\1千字节'),
    (re.compile(r'(\d+)\s*TB(?![a-zA-Z])'), r'\1太字节'),
    # 长度/重量
    (re.compile(r'(\d+)\s*mm(?![a-zA-Z])'), r'\1毫米'),
    (re.compile(r'(\d+)\s*cm(?![a-zA-Z])'), r'\1厘米'),
    (re.compile(r'(\d+)\s*km(?![a-zA-Z])'), r'\1公里'),
    (re.compile(r'(\d+)\s*kg(?![a-zA-Z])'), r'\1公斤'),
    (re.compile(r'(\d+)\s*mg(?![a-zA-Z])'), r'\1毫克'),
    # 温度
    (re.compile(r'(\d+)\s*°C(?![a-zA-Z])'), r'\1摄氏度'),
    (re.compile(r'(\d+)\s*°F(?![a-zA-Z])'), r'\1华氏度'),
    # 网络
    (re.compile(r'(\d+)\s*Mbps(?![a-zA-Z])'), r'\1兆比特每秒'),
    (re.compile(r'(\d+)\s*Gbps(?![a-zA-Z])'), r'\1吉比特每秒'),
    (re.compile(r'(\d+)\s*Kbps(?![a-zA-Z])'), r'\1千比特每秒'),
]

def _preprocess_text(text: str) -> str:
    """合成前文本预处理：品牌名逐位读 + 单位符号转中文。"""
    from .spoken_text import normalize_technical_text
    text = normalize_technical_text(text)
    for brand, reading in _BRAND_MAP.items():
        # Disambiguate the brand from prices, capacities and model numbers.
        text = re.sub(r'(?<![A-Za-z0-9])' + re.escape(brand)
                      + r'(?=\s*(?:安全|浏览器|杀毒|卫士|公司|集团|品牌))', reading, text)
    for pattern, replacement in _UNIT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

# 空闲超时（秒）：超时后自动卸载模型，释放显存
IDLE_TIMEOUT = int(os.getenv("TTS_IDLE_TIMEOUT", "600"))

# 模型缓存根目录
CACHE_DIR = Path(os.getenv("TTS_MODEL_DIR", str(Path.home() / ".cache/hermes-speech/tts")))

# ---------------------------------------------------------------------------
# 引擎加载
# ---------------------------------------------------------------------------

_runtime = ModelRuntime()
_engine = None


def _get_engine():
    """获取/初始化当前引擎（单例）"""
    global _engine
    if _engine is not None:
        return _engine

    from .engines import get_engine_class, list_engines

    logger.info(f"Initializing engine: {TTS_ENGINE}")
    engine_cls = get_engine_class(TTS_ENGINE)
    _engine = engine_cls(cache_dir=CACHE_DIR)

    # Register prompt_text for zero-shot voice cloning.
    # When a voice file matches a registered path, inference_zero_shot is
    # used instead of inference_cross_lingual — better quality, needs the
    # transcript of the prompt audio.
    _prompt_dir = os.path.join(os.path.dirname(__file__), "models")
    _feiying_prompt = os.path.join(_prompt_dir, "feiying_prompt_zero_shot_16k.wav")
    _feiying_text = "瞬即逝，也能成为心底的永恒。"
    if os.path.isfile(_feiying_prompt):
        _engine._register_prompt_text(_feiying_prompt, _feiying_text)
        logger.info(f"Registered prompt_text for feiying voice: {_feiying_prompt}")

    return _engine


def _resolve_model() -> str:
    """确定使用的模型名"""
    if TTS_MODEL:
        return TTS_MODEL
    engine = _get_engine()
    return engine.info().default_model


def _resolve_requested_model(model):
    # Explicit requests keep their identity; the engine validates compatibility.
    return model or _resolve_model()


# ---------------------------------------------------------------------------
# 空闲超时 —— 超时后自动卸载模型释放显存
# ---------------------------------------------------------------------------



def _touch_activity():
    _runtime.last_activity = time.monotonic()


def _get_idle_seconds():
    return _runtime.idle_seconds


def _idle_monitor(stop):
    while not stop.wait(30):
        if _engine is not None and _runtime.unload_if_idle(_engine, IDLE_TIMEOUT):
            logger.info("Model unloaded (idle timeout)")



# ---------------------------------------------------------------------------
# FastAPI 生命周期
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_cpu_policy()
    logger.info("TTS Service starting...")
    logger.info(f"Engine: {TTS_ENGINE}, Default voice: {TTS_DEFAULT_VOICE}")
    logger.info(f"Cache dir: {CACHE_DIR}")
    logger.info(f"Idle timeout: {IDLE_TIMEOUT}s")

    if os.getenv("SPEECH_PRELOAD", "0") == "1":
        try:
            engine = _get_engine()
            model = _resolve_model()
            logger.info(f"Loading model: {model}")
            await run_in_threadpool(_prepare_model, engine, model)
            _touch_activity()
            logger.info(f"✅ TTS Service ready (engine={TTS_ENGINE}, model={model})")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")

    stop = threading.Event()
    monitor_thread = threading.Thread(target=_idle_monitor, args=(stop,), daemon=True, name="speech-idle-monitor")
    monitor_thread.start()

    try:
        yield
    finally:
        stop.set()
        monitor_thread.join(timeout=2)
    logger.info("TTS Service shutting down...")


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------

app = FastAPI(
    title="公共 TTS 服务",
    description="文字转语音 HTTP API - 多引擎支持 (cosyvoice2 / cosyvoice3)",
    version="1.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

class SynthesizeResponse(BaseModel):
    success: bool
    audio_path: str = ""
    provider: str = "local"
    model: str
    engine: str
    voice: str
    duration_ms: Optional[int] = None
    processing_time_ms: Optional[int] = None
    sample_rate: Optional[int] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    engine: str
    model_loaded: bool
    model_name: Optional[str]
    idle_seconds: float = 0
    version: str
    warmed: bool = False
    active: bool = False


class ModelsResponse(BaseModel):
    available: list
    current: str


class EnginesResponse(BaseModel):
    available: list
    current: str


class VoicesResponse(BaseModel):
    available: list
    current: str


# ---------------------------------------------------------------------------
# API 端点
# ---------------------------------------------------------------------------

def _prepare_model(engine, model):
    with _runtime.request():
        started = time.perf_counter()
        engine.load_model(model)
        logger.info("timing startup_load_ms=%.2f", (time.perf_counter() - started) * 1000)


@app.get("/ready")
def readiness():
    state = readiness_state()
    ready = state["ready"]
    return JSONResponse(status_code=200 if ready else 503,
                        content=state)


def readiness_state():
    loaded = bool(_engine is not None and _engine.model_loaded)
    return {"ready": loaded and _runtime.warmed, "model_loaded": loaded,
            "warmed": _runtime.warmed, "active": _runtime.active}


@app.post("/warmup")
def warmup():
    with _runtime.request() as timings:
        engine = _get_engine()
        started = time.perf_counter()
        engine.load_model(_resolve_model())
        timings["load_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if _runtime.warmed:
            return {"ready": True, "already_warmed": True, "timings": timings}
        started = time.perf_counter()
        chunks = engine.synthesize_chunks("你好，语音服务已准备好。", voice=TTS_DEFAULT_VOICE)
        count = sum(1 for item in chunks if item["speech"] is not None and len(item["speech"]))
        if not count:
            raise RuntimeError("Warmup produced no audio")
        _runtime.warmed = True
        timings["warmup_ms"] = round((time.perf_counter() - started) * 1000, 2)
        logger.info("timing warmup=%s", timings)
        return {"ready": True, "already_warmed": False, "timings": timings}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    engine = _get_engine()
    return HealthResponse(
        status="healthy" if engine.model_loaded else "degraded",
        engine=TTS_ENGINE,
        model_loaded=engine.model_loaded,
        model_name=engine.model_name,
        idle_seconds=_get_idle_seconds(),
        warmed=_runtime.warmed,
        active=_runtime.active,
        version="1.1.0",
    )


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    engine = _get_engine()
    return ModelsResponse(
        available=engine.list_available_models(),
        current=engine.model_name or _resolve_model(),
    )


@app.get("/engines", response_model=EnginesResponse)
async def list_engines():
    from .engines import list_engines as _list_engines
    return EnginesResponse(
        available=_list_engines(),
        current=TTS_ENGINE,
    )


@app.get("/voices", response_model=VoicesResponse)
async def list_voices():
    engine = _get_engine()
    return VoicesResponse(
        available=engine.list_voices(),
        current=TTS_DEFAULT_VOICE,
    )


@app.post("/synthesize")
def synthesize_audio(
    text: str = Form(..., description="要合成的文本"),
    voice: Optional[str] = Form(None, description="音色名称"),
    model: Optional[str] = Form(None, description="模型名称"),
    speed: Optional[float] = Form(None, description="语速倍率 (1.0=正常)"),
    language: Optional[str] = Form(None, description="语言代码"),
    format: str = Form("wav", description="输出格式 (wav/mp3/ogg)"),
    prompt_text: Optional[str] = Form(None, description="参考音频对应的文字（zero-shot克隆用）"),
):
    """
    合成语音

    - **text**: 要合成的文本（最长 5000 字符）
    - **voice**: 音色名称（默认 中文女）
    - **model**: 模型名称（可选，默认使用服务配置的模型）
    - **speed**: 语速倍率 (1.0=正常)
    - **language**: 语言代码
    - **format**: 输出音频格式 (wav, mp3, ogg)

    返回音频文件。
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long: {len(text)} chars (max {MAX_TEXT_LENGTH})",
        )

    start_time = time.perf_counter()
    temp_path = None

    try:
        with _runtime.request() as timings:
            engine = _get_engine()
            _touch_activity()

            use_model = _resolve_requested_model(model)
            try:
                load_start = time.perf_counter()
                engine.load_model(use_model)
                timings["load_ms"] = round((time.perf_counter() - load_start) * 1000, 2)
            except Exception as e:
                logger.error(f"Model loading failed: {e}")
                return JSONResponse(
                    status_code=500,
                    content=SynthesizeResponse(
                        success=False,
                        model=use_model,
                        engine=TTS_ENGINE,
                        voice=voice or TTS_DEFAULT_VOICE,
                        error=f"Model loading failed: {e}",
                    ).dict(),
                )

            use_voice = voice or TTS_DEFAULT_VOICE
            use_language = language or TTS_DEFAULT_LANGUAGE

            # 生成临时文件
            ext = format if format in ("wav", "mp3", "ogg", "flac") else "wav"
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                temp_path = tmp.name

            logger.info(f"Synthesizing: {len(text)} chars, voice={use_voice} [engine={TTS_ENGINE}]")

            # 文本预处理：品牌名 + 单位符号归一化
            text = _preprocess_text(text)

            # Register prompt_text for zero-shot cloning if provided
            if prompt_text and use_voice and os.path.isfile(use_voice):
                engine._register_prompt_text(use_voice, prompt_text)
                logger.info(f"Registered prompt_text for voice={use_voice}: {prompt_text[:30]}...")
            else:
                logger.info(f"No prompt_text registered: prompt_text={bool(prompt_text)}, use_voice={use_voice}, isfile={os.path.isfile(use_voice) if use_voice else 'N/A'}")

            result = engine.synthesize(
                text=text,
                output_path=temp_path,
                voice=use_voice,
                speed=speed,
                language=use_language,
            )

            if result.error:
                return JSONResponse(
                    status_code=500,
                    content=SynthesizeResponse(
                        success=False,
                        model=use_model,
                        engine=TTS_ENGINE,
                        voice=use_voice,
                        error=result.error,
                        processing_time_ms=result.processing_time_ms,
                    ).dict(),
                )

            logger.info(
                f"Synthesis complete: {len(text)} chars -> "
                f"{result.duration_ms}ms audio (processing: {result.processing_time_ms}ms)"
            )

            _runtime.warmed = True
            timings["total_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info("timing synthesize=%s", timings)

            # 返回音频文件
            return FileResponse(
                path=result.audio_path,
                media_type=f"audio/{ext}",
                filename=f"tts_output.{ext}",
            )

    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Synthesis failed: {e}",
                "engine": TTS_ENGINE,
            },
        )
    finally:
        # 延迟清理临时文件（FileResponse 发送完后）
        if temp_path and Path(temp_path).exists():
            # FileResponse 会在后台发送，不能立即删除
            # 注册一个延迟清理
            def _cleanup():
                time.sleep(30)  # 等待文件发送完毕
                try:
                    Path(temp_path).unlink()
                except Exception:
                    pass
            threading.Thread(target=_cleanup, daemon=True).start()


# ---------------------------------------------------------------------------
# 流式合成端点 (SSE) — 模型 chunk 生成一块就推一块，首包 = 第一个 chunk 产出时刻
# ---------------------------------------------------------------------------

def _float32_to_pcm16_bytes(speech) -> bytes:
    """Convert a mono float32 sample array (values ~[-1, 1]) to int16 LE PCM."""
    import numpy as np

    arr = np.asarray(speech, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.squeeze()
    if arr.ndim != 1:
        arr = arr.flatten()
    pcm = np.clip(arr * 32767.0, -32768.0, 32767.0).astype("<i2")
    return pcm.tobytes()


_stream_requests = {}


@app.post("/cancel/{request_id}")
async def cancel_stream(request_id: str):
    cancelled = _stream_requests.get(request_id)
    if cancelled is not None:
        cancelled.set()
    return {"request_id": request_id, "cancelled": cancelled is not None}


@app.post("/synthesize-stream")
async def synthesize_stream(
    text: str = Form(..., description="要合成的文本"),
    voice: Optional[str] = Form(None, description="音色/参考音频路径"),
    model: Optional[str] = Form(None, description="模型名称"),
    speed: Optional[float] = Form(None, description="语速倍率 (1.0=正常)"),
    language: Optional[str] = Form(None, description="语言代码"),
    prompt_text: Optional[str] = Form(None, description="参考音频对应的文字（zero-shot克隆用）"),
    request_id: Optional[str] = Form(None),
):
    """SSE 流式合成：文本 -> 每生成一块音频就推一个事件。

    与 /synthesize 完全相同的文本预处理 / 音色 / prompt_text 语义，
    只是不再等整句合成完 — 事件流:

      data: {"type":"start","voice":...,"model":...,"sr":24000}
      data: {"type":"audio","idx":0,"sr":24000,"ms":N,"b64":"<int16 LE PCM base64>"}
      data: {"type":"audio","idx":1,...}
      data: {"type":"done","chunks":K,"total_ms":M}

    出错时推: data: {"type":"error","message":"..."}（HTTP 200，事件级错误），
    或模型加载失败等早期错误直接返回 HTTP 错误 JSON（与 /synthesize 一致）。
    音频为 16-bit 小端单声道 PCM，采样率见事件内 sr（CosyVoice2 = 24000）。
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long: {len(text)} chars (max {MAX_TEXT_LENGTH})",
        )

    use_model = _resolve_requested_model(model)
    use_voice = voice or TTS_DEFAULT_VOICE
    use_language = language or TTS_DEFAULT_LANGUAGE
    text = _preprocess_text(text)
    request_start = time.perf_counter()
    request_id = request_id if isinstance(request_id, str) else uuid4().hex
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", request_id):
        raise HTTPException(status_code=400, detail="Invalid request_id")
    if request_id in _stream_requests:
        raise HTTPException(status_code=409, detail="Request already active")
    cancelled = threading.Event()
    _stream_requests[request_id] = cancelled

    def _event(payload: dict) -> str:
        return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    def _event_gen():
        from .diagnostics import RequestTrace
        with _runtime.request() as timings:
            if cancelled.is_set():
                return
            chunk_count = 0
            total_audio_ms = 0
            first_audio_sent = False
            engine = None
            trace = RequestTrace(request_id)
            trace.emit("acquired", queue_ms=timings["queue_ms"])
            trace.start_resources()
            try:
                engine = _get_engine()
                load_start = time.perf_counter()
                with trace.stage("load"):
                    engine.load_model(use_model)
                engine._trace = trace
                if cancelled.is_set():
                    return
                engine._cancel_event = cancelled
                timings["load_ms"] = round((time.perf_counter() - load_start) * 1000, 2)
                if prompt_text and os.path.isfile(use_voice):
                    engine._register_prompt_text(use_voice, prompt_text)
                yield _event({
                    "type": "start",
                    "request_id": request_id,
                    "voice": use_voice,
                    "model": use_model,
                    "sr": 24000,
                })
                with closing(engine.synthesize_chunks(
                    text,
                    voice=use_voice,
                    speed=speed,
                    language=use_language,
                )) as chunks:
                    for item in chunks:
                        if cancelled.is_set():
                            break
                        speech = item["speech"]
                        sr = int(item["sample_rate"])
                        if speech is None or len(speech) == 0:
                            continue
                        with trace.stage("pcm_encode"):
                            pcm = _float32_to_pcm16_bytes(speech)
                        if not pcm:
                            continue
                        ms = int(len(pcm) / (sr * 2) * 1000)
                        total_audio_ms += ms
                        chunk_count += 1
                        if not first_audio_sent:
                            timings["first_pcm_ms"] = round((time.perf_counter() - request_start) * 1000, 2)
                        first_audio_sent = True
                        with trace.stage("sse_encode"):
                            event = _event({
                                "type": "audio",
                                "idx": item["idx"],
                                "sr": sr,
                                "ms": ms,
                                "b64": base64.b64encode(pcm).decode("ascii"),
                            })
                        trace.checkpoint("chunk", index=chunk_count, audio_ms=ms)
                        with trace.stage("downstream_wait"):
                            yield event
                if cancelled.is_set():
                    return
                _runtime.warmed = first_audio_sent
                timings["total_ms"] = round((time.perf_counter() - request_start) * 1000, 2)
                logger.info("timing synthesize_stream=%s", timings)
                yield _event({
                    "timings": timings,
                    "type": "done",
                    "chunks": chunk_count,
                    "total_ms": total_audio_ms,
                    "audio_sent": first_audio_sent,
                })
            except Exception as e:
                logger.error(f"Streaming synthesis failed: {e}", exc_info=True)
                yield _event({"type": "error", "message": str(e)})
            finally:
                if engine is not None:
                    engine._cancel_event = None
                    engine._trace = None
                trace.close(cancelled=cancelled.is_set())
                logger.info("stream_finished request_id=%s cancelled=%s elapsed_ms=%.2f",
                            request_id, cancelled.is_set(), (time.perf_counter() - request_start) * 1000)

    logger.info(f"Streaming synthesize: {len(text)} chars, voice={use_voice} [engine={TTS_ENGINE}]")
    return StreamingResponse(
        stream_events(_event_gen(), cancelled, lambda: _stream_requests.pop(request_id, None)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# 错误处理
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc)},
    )


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("公共 TTS 服务 v1.0")
    logger.info("=" * 60)
    logger.info(f"Host: {SERVICE_HOST}")
    logger.info(f"Port: {SERVICE_PORT}")
    logger.info(f"Engine: {TTS_ENGINE}")
    logger.info(f"Model: {TTS_MODEL or '(default)'}")
    logger.info(f"Default voice: {TTS_DEFAULT_VOICE}")
    logger.info(f"Max text length: {MAX_TEXT_LENGTH}")
    logger.info("=" * 60)

    uvicorn.run(
        app,
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        workers=WORKERS,
    )
