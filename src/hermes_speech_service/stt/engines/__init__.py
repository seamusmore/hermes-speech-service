"""
STT 引擎抽象层

模板方法模式：
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

logger = logging.getLogger("stt-service")


@dataclass
class TranscribeResult:
    text: str
    language: str = ""
    duration_ms: Optional[int] = None
    processing_time_ms: Optional[int] = None
    emotion: Optional[str] = None
    event: Optional[str] = None


@dataclass
class EngineInfo:
    name: str
    display_name: str
    models: list[str] = field(default_factory=list)
    default_model: str = ""
    supports_language_param: bool = True
    supports_streaming: bool = False


class BaseEngine(ABC):
    """STT 引擎基类 — 模板方法模式

    _ensure_model_loaded() 定义统一流程：
      已加载 → 跳过
      缓存检查 (_check_model_cached)
      缺失 → 自动下载 (_auto_download)
      加载 (_load_model)
    """

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
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> TranscribeResult:
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

    def health_detail(self) -> dict:
        return {
            "engine": self.info().name,
            "model_loaded": self.model_loaded,
            "model_name": self._model_name,
        }

    def unload(self) -> None:
        """卸载模型释放内存。子类有特殊清理需求时可 override。"""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_name = None
            gc.collect()
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
        _lazy_import_engines(name)
    if name not in _ENGINE_CLASSES:
        raise ValueError(f"Unknown engine: {name}. Available: {list(_ENGINE_CLASSES.keys())}")
    return _ENGINE_CLASSES[name]


def list_engines() -> list[str]:
    _lazy_import_engines()
    return list(_ENGINE_CLASSES.keys())


def _lazy_import_engines(name=None):
    import os as _os
    engine = name or _os.environ.get("STT_ENGINE", "sensevoice")
    _IMPORT_MAP = {
        "whisper": ".whisper_engine",
        "sensevoice": ".sensevoice_torch",
        "sensevoice-q8": ".sensevoice_q8",
    }
    mod = _IMPORT_MAP.get(engine)
    if mod:
        import importlib
        importlib.import_module(mod, package=__package__)
    else:
        from . import whisper_engine  # noqa: F401
        from . import sensevoice_torch  # noqa: F401
        from . import sensevoice_q8  # noqa: F401

