"""
faster-whisper 引擎实现

基于 Systran/faster-whisper，支持 tiny/base/small/medium/large。
覆盖 _auto_download() 走 HF 镜像，其余流程由基类模板方法 _ensure_model_loaded 统一管理。
"""

from __future__ import annotations
import os
import time
import logging
from pathlib import Path
from typing import Optional

from . import BaseEngine, TranscribeResult, EngineInfo, register_engine

logger = logging.getLogger("stt-service")


@register_engine("whisper")
class WhisperEngine(BaseEngine):

    MODEL_MAP = {
        "tiny":   "Systran/faster-whisper-tiny",
        "base":   "Systran/faster-whisper-base",
        "small":  "Systran/faster-whisper-small",
        "medium": "Systran/faster-whisper-medium",
        "large":  "Systran/faster-whisper-large-v3",
    }

    LEGACY_CACHE_DIR = Path("/mnt/stt-service/models/whisper")

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="whisper",
            display_name="faster-whisper (Whisper)",
            models=list(self.MODEL_MAP.keys()),
            default_model="base",
        )

    def _resolve_download_root(self) -> str:
        for d in [self.LEGACY_CACHE_DIR, self.cache_dir]:
            if d.exists():
                return str(d)
        return str(self.LEGACY_CACHE_DIR)

    def _full_name(self) -> str:
        return self.MODEL_MAP.get(self._model_name or self.info().default_model,
                                  self._model_name or self.info().default_model)

    def _check_model_cached(self) -> bool:
        full_name = self._full_name()
        repo_dir = f"models--{full_name.replace('/', '--')}"
        for d in [self.LEGACY_CACHE_DIR, self.cache_dir]:
            if (d / repo_dir).exists():
                return True
        return False

    def _auto_download(self) -> None:
        full_name = self._full_name()
        logger.info("[whisper] Auto-downloading from HF mirror: %s", full_name)

        from faster_whisper import WhisperModel

        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        old_offline = os.environ.get("HF_HUB_OFFLINE", "1")
        os.environ["HF_HUB_OFFLINE"] = "0"

        try:
            download_root = self._resolve_download_root()
            Path(download_root).mkdir(parents=True, exist_ok=True)
            tmp_model = WhisperModel(
                full_name, device="cpu", compute_type="int8",
                download_root=download_root, local_files_only=False,
            )
            del tmp_model
        finally:
            os.environ["HF_HUB_OFFLINE"] = old_offline

        if not self._check_model_cached():
            raise FileNotFoundError(f"Whisper model download failed: {full_name}")
        logger.info("[whisper] Auto-download complete")

    def _load_model(self) -> None:
        from faster_whisper import WhisperModel

        full_name = self._full_name()
        download_root = self._resolve_download_root()
        Path(download_root).mkdir(parents=True, exist_ok=True)

        logger.info("[whisper] Loading model from cache: %s", full_name)
        start = time.time()

        old_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        try:
            self._model = WhisperModel(
                full_name, device="cpu", compute_type="int8",
                download_root=download_root, local_files_only=True,
            )
        finally:
            os.environ["HF_ENDPOINT"] = old_endpoint

        elapsed = time.time() - start
        logger.info("[whisper] Model loaded in %.2fs", elapsed)

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> TranscribeResult:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        start = time.time()
        kwargs = {"beam_size": 5}
        if language and language != "auto":
            kwargs["language"] = language

        segments, info = self._model.transcribe(audio_path, **kwargs)
        text = " ".join(seg.text.strip() for seg in segments)

        return TranscribeResult(
            text=text,
            language=language or getattr(info, "language", "auto"),
            duration_ms=int(info.duration * 1000) if hasattr(info, "duration") else None,
            processing_time_ms=int((time.time() - start) * 1000),
        )

