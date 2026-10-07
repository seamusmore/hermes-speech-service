"""
SenseVoice Q8 引擎实现（sherpa-onnx + INT8 量化 ONNX）

model.int8.onnx 为 INT8 量化权重（~228MB），通过 sherpa-onnx → onnxruntime 推理。
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


def _patch_sherpa_onnx_mmap():
    import sherpa_onnx
    _original = sherpa_onnx.OfflineRecognizer.from_sense_voice

    @classmethod
    def _patched(cls, model, tokens, num_threads=1, **kwargs):
        import onnxruntime as ort
        _orig_init = ort.InferenceSession.__init__

        def _mmap_init(self, path_or_bytes, sess_options=None,
                       providers=None, provider_options=None, **ikwargs):
            if sess_options is None:
                sess_options = ort.SessionOptions()
            sess_options.enable_mmap = True
            _orig_init(self, path_or_bytes,
                       sess_options=sess_options,
                       providers=providers,
                       provider_options=provider_options,
                       **ikwargs)

        ort.InferenceSession.__init__ = _mmap_init
        try:
            return _original.__func__(cls, model, tokens, num_threads, **kwargs)
        finally:
            ort.InferenceSession.__init__ = _orig_init

    sherpa_onnx.OfflineRecognizer.from_sense_voice = _patched
    return _original


def _restore_sherpa_onnx(_original):
    import sherpa_onnx
    sherpa_onnx.OfflineRecognizer.from_sense_voice = _original


@register_engine("sensevoice-q8")
class SenseVoiceQ8Engine(BaseEngine):

    MODEL_DIR = "sensevoice-q8"
    MODEL_REPO = "poloniumrock/SenseVoiceSmallOnnx"

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="sensevoice-q8",
            display_name="SenseVoice-Small Q8 (sherpa-onnx INT8)",
            models=["sensevoice-q8"],
            default_model="sensevoice-q8",
            supports_language_param=False,
            supports_streaming=False,
        )

    def _check_model_cached(self) -> bool:
        return (self.cache_dir / self.MODEL_DIR / "model.int8.onnx").exists()

    def _auto_download(self) -> None:
        """从 ModelScope 下载 SenseVoiceSmallOnnx 并展平到缓存目录。"""
        from modelscope.hub.snapshot_download import snapshot_download

        model_dir = self.cache_dir / self.MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[sensevoice-q8] Auto-downloading from ModelScope: %s", self.MODEL_REPO)

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
                        logger.info("[sensevoice-q8] Copied: %s", f.name)
            else:
                for root, dirs, files in os.walk(tmp_dir):
                    for f in files:
                        src_file = Path(root) / f
                        if not (model_dir / f).exists():
                            shutil.copy2(str(src_file), str(model_dir / f))
                            logger.info("[sensevoice-q8] Copied: %s", f)

        if not self._check_model_cached():
            raise FileNotFoundError(
                f"Auto-download completed but model.int8.onnx not found in {model_dir}"
            )
        logger.info("[sensevoice-q8] Auto-download complete")

    def _load_model(self) -> None:
        import sherpa_onnx

        model_dir = self.cache_dir / self.MODEL_DIR
        model_path = str(model_dir / "model.int8.onnx")
        tokens_path = str(model_dir / "tokens.txt")

        logger.info("[sensevoice-q8] Loading from %s", model_path)

        _orig = _patch_sherpa_onnx_mmap()
        try:
            self._model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=model_path,
                tokens=tokens_path,
                num_threads=2,
                use_itn=True,
                language="auto",
            )
        finally:
            _restore_sherpa_onnx(_orig)

        self._model_name = "sensevoice-q8"
        logger.info("[sensevoice-q8] Model ready (INT8 mmap, ~228MB)")

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> TranscribeResult:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        import soundfile as sf
        start = time.time()

        samples, sr = sf.read(audio_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples[:, 0]

        stream = self._model.create_stream()
        stream.accept_waveform(sr, samples)
        self._model.decode_stream(stream)
        processing_ms = int((time.time() - start) * 1000)

        raw_text = stream.result.text
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

