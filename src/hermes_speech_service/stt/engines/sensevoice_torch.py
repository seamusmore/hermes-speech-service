"""
SenseVoice 引擎实现（funasr + torch FP32）

model.pt 为 FP32 权重（893MB），通过 mmap 懒加载，RSS ~530MB。
"""

from __future__ import annotations
import os
import shutil
import tempfile
import time
import re
import gc
import logging
from pathlib import Path
from typing import Optional

from . import BaseEngine, TranscribeResult, EngineInfo, register_engine

logger = logging.getLogger("stt-service")
_RICH_TAG_RE = re.compile(r"<\|[^|]*\|>")


def _patch_funasr_loader():
    import torch
    import funasr.auto.auto_model as auto_mod
    _original = auto_mod.load_pretrained_model

    def _mmap_loader(path, model, ignore_init_mismatch=True,
                     map_location="cpu", oss_bucket=None,
                     scope_map=None, excludes=None, **kwargs):
        checkpoint = torch.load(path, map_location=map_location, mmap=True)
        src_state = checkpoint.get("state_dict",
                    checkpoint.get("model_state_dict",
                    checkpoint.get("model", checkpoint)))
        if isinstance(scope_map, str):
            scope_map = scope_map.split(",")
        scope_map = (scope_map or []) + ["module.", "None"]
        if excludes is not None and isinstance(excludes, str):
            excludes = excludes.split(",")
        matched = 0
        for name, param in model.named_parameters():
            if excludes and any(name.startswith(ex) for ex in excludes):
                continue
            k_src = name
            for i in range(0, len(scope_map), 2):
                sp = scope_map[i] if scope_map[i].lower() != "none" else ""
                dp = scope_map[i + 1] if scope_map[i + 1].lower() != "none" else ""
                if dp == "" and (sp + name) in src_state:
                    k_src = sp + name
                elif name.startswith(dp) and name.replace(dp, sp, 1) in src_state:
                    k_src = name.replace(dp, sp, 1)
            if k_src in src_state:
                src_tensor = src_state[k_src]
                if ignore_init_mismatch and param.shape != src_tensor.shape:
                    continue
                param.data = src_tensor
                matched += 1
        logger.info("[sensevoice] Loading ckpt: %s, matched=%d params (mmap)", path, matched)

    auto_mod.load_pretrained_model = _mmap_loader
    return _original


def _restore_funasr_loader(original):
    import funasr.auto.auto_model as auto_mod
    auto_mod.load_pretrained_model = original


@register_engine("sensevoice")
class SenseVoiceEngine(BaseEngine):

    MODEL_DIR = "sensevoice"
    MODEL_REPO = "iic/SenseVoiceSmall"

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="sensevoice",
            display_name="SenseVoice-Small (funasr+torch)",
            models=["sensevoice"],
            default_model="sensevoice",
            supports_language_param=False,
            supports_streaming=False,
        )

    def _check_model_cached(self) -> bool:
        return (self.cache_dir / self.MODEL_DIR / "model.pt").exists()

    def _auto_download(self) -> None:
        """从 ModelScope 下载 SenseVoiceSmall 并展平到缓存目录。"""
        from modelscope.hub.snapshot_download import snapshot_download

        model_dir = self.cache_dir / self.MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[sensevoice] Auto-downloading from ModelScope: %s", self.MODEL_REPO)

        with tempfile.TemporaryDirectory() as tmp_dir:
            downloaded = snapshot_download(self.MODEL_REPO, cache_dir=tmp_dir)
            src = Path(downloaded)
            if src.exists():
                for f in src.iterdir():
                    dst = model_dir / f.name
                    if not dst.exists():
                        if f.is_dir():
                            shutil.copytree(str(f), str(dst))
                        else:
                            shutil.copy2(str(f), str(dst))
                        logger.info("[sensevoice] Copied: %s", f.name)
            else:
                for root, dirs, files in os.walk(tmp_dir):
                    for f in files:
                        src_file = Path(root) / f
                        if not (model_dir / f).exists():
                            shutil.copy2(str(src_file), str(model_dir / f))
                            logger.info("[sensevoice] Copied: %s", f)

        if not self._check_model_cached():
            raise FileNotFoundError(
                f"Auto-download completed but model.pt not found in {model_dir}"
            )
        logger.info("[sensevoice] Auto-download complete")

    def _load_model(self) -> None:
        from funasr import AutoModel

        model_dir = str(self.cache_dir / self.MODEL_DIR)
        logger.info("[sensevoice] Loading from %s (mmap)", model_dir)

        _orig_loader = _patch_funasr_loader()
        try:
            device = os.getenv("STT_DEVICE", "cuda")
            logger.info("[sensevoice] Using device: %s", device)
            self._model = AutoModel(
                model=model_dir,
                disable_update=True,
                device=device,
                ncpu=2 if device == "cpu" else 0,
            )
        finally:
            _restore_funasr_loader(_orig_loader)

        self._model_name = "sensevoice"
        logger.info("[sensevoice] Model ready")

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> TranscribeResult:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        start = time.time()
        result = self._model.generate(input=audio_path, language=language or "auto", use_itn=True)
        processing_ms = int((time.time() - start) * 1000)

        if not result:
            return TranscribeResult(text="", language=language or "auto", processing_time_ms=processing_ms)

        item = result[0]
        raw_text = item.get("text", "")
        emotion = None
        emo_match = re.search(r"<\|EMO_(\w+)\|>", raw_text)
        if emo_match:
            emotion = emo_match.group(1).lower()
        event_tags = re.findall(r"<\|((?!EMO_)[A-Z_]+)\|>", raw_text)
        event = ",".join(event_tags).lower() if event_tags else None

        return TranscribeResult(
            text=_RICH_TAG_RE.sub("", raw_text).strip(),
            language=language or "auto",
            processing_time_ms=processing_ms,
            emotion=emotion,
            event=event,
        )

