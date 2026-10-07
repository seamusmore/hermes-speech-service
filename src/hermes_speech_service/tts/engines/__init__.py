"""
TTS 引擎抽象层

对称仿照 hermes-stt-service/engines/__init__.py 的模板方法模式：
- _ensure_model_loaded()  基类统一入口
- _check_model_cached()   子类告知缓存是否存在
- _auto_download()         子类实现下载（抽象）
- _load_model()            子类实现具体加载（抽象）
"""

from __future__ import annotations
import os
import gc
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("tts-service")


@dataclass
class SynthesizeResult:
    audio_path: str
    processing_time_ms: Optional[int] = None
    sample_rate: Optional[int] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None


@dataclass
class EngineInfo:
    name: str
    display_name: str
    models: list[str] = field(default_factory=list)
    default_model: str = ""
    supports_streaming: bool = False
    supports_voice_cloning: bool = False
    voices: list[str] = field(default_factory=list)


class BaseEngine(ABC):
    """TTS 引擎基类 — 模板方法模式（对称 STT）"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self._model = None
        self._model_name: Optional[str] = None

    # ---- 抽象方法 ----

    @abstractmethod
    def info(self) -> EngineInfo:
        ...

    @abstractmethod
    def _check_model_cached(self) -> bool:
        ...

    @abstractmethod
    def _auto_download(self) -> None:
        """下载模型文件到缓存。每个子类自己决定从哪下载、怎么下载。"""
        ...

    @abstractmethod
    def _load_model(self) -> None:
        ...

    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        language: Optional[str] = None,
    ) -> SynthesizeResult:
        ...

    # ---- 模板方法 ----

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        if not self._check_model_cached():
            self._auto_download()
        self._load_model()

    def load_model(self, model_name: str) -> None:
        if self._model is not None and self._model_name == model_name:
            return
        self._model_name = model_name
        self._ensure_model_loaded()

    # ---- 属性 ----

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    def list_available_models(self) -> list[str]:
        return self.info().models

    def list_voices(self) -> list[str]:
        return self.info().voices

    def health_detail(self) -> dict:
        return {
            "engine": self.info().name,
            "model_loaded": self.model_loaded,
            "model_name": self._model_name,
        }

    def unload(self) -> None:
        """卸载模型释放显存/内存。"""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_name = None
            gc.collect()
            # 尝试清空 CUDA 缓存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("[%s] Model unloaded", self.info().name)


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

_ENGINE_CLASSES: dict[str, type[BaseEngine]] = {}


def register_engine(name: str):
    def wrapper(cls: type[BaseEngine]):
        _ENGINE_CLASSES[name] = cls
        return cls
    return wrapper


def get_engine_class(name: str) -> type[BaseEngine]:
    if name not in _ENGINE_CLASSES:
        _lazy_import_engines()
    if name not in _ENGINE_CLASSES:
        raise ValueError(f"Unknown engine: {name}. Available: {list(_ENGINE_CLASSES.keys())}")
    return _ENGINE_CLASSES[name]


def list_engines() -> list[str]:
    _lazy_import_engines()
    return list(_ENGINE_CLASSES.keys())


def _lazy_import_engines():
    # Adapters import model dependencies only when load_model is called.
    from . import cosyvoice2, cosyvoice3  # noqa: F401

