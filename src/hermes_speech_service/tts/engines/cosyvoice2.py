"""
CosyVoice2 引擎实现

使用 CosyVoice2-0.5B 模型，支持 SFT 预设音色和零样本声音克隆。
模型从 ModelScope 下载，GPU 推理（CUDA）。

依赖：
- torch + torchaudio (CUDA)
- cosyvoice (从 GitHub 安装)
- modelscope (下载模型)
- numpy
- soundfile

环境变量：
- TTS_DEVICE: cuda | cpu (默认 cuda)
- TTS_MODEL_DIR: 模型缓存目录（默认 <service>/models）
"""

from __future__ import annotations
from contextlib import nullcontext
from ..diagnostics import measured_chunks, onset_metrics
import os
import shutil
import tempfile
import time
import logging
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from . import BaseEngine, SynthesizeResult, EngineInfo, register_engine

logger = logging.getLogger("tts-service")

# CosyVoice2 预设音色（SFT 模型内置）
_SFT_VOICES = ["中文女", "中文男", "英语女", "英语男", "日语女", "日语男", "韩语女", "韩语男"]


@register_engine("cosyvoice2")
class CosyVoice2Engine(BaseEngine):

    MODEL_DIR = "CosyVoice2-0.5B"
    MODEL_REPO = "iic/CosyVoice2-0.5B"
    MODEL_NAME = "cosyvoice2"
    MODEL_CLASS = "CosyVoice2"
    CONFIG_FILE = "cosyvoice2.yaml"

    def __init__(self, cache_dir: Path):
        super().__init__(cache_dir)
        self._device = os.getenv("TTS_DEVICE", "cuda")
        self._PROMPT_TEXTS = {}
        self._voice_cache = OrderedDict()
        self._cancel_event = None
        self._token_cancel_llm = None
        self._token_cancel_wrappers = {}

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="cosyvoice2",
            display_name="CosyVoice2-0.5B (阿里通义)",
            models=["cosyvoice2"],
            default_model="cosyvoice2",
            supports_streaming=True,
            supports_voice_cloning=True,
            voices=_SFT_VOICES,
        )

    def load_model(self, model_name: str) -> None:
        if model_name != self.MODEL_NAME:
            raise ValueError(
                f'This process uses {self.MODEL_NAME}; requested {model_name!r}. '
                f'Use model={self.MODEL_NAME} or omit the model field. '
                'Restart the service with TTS_ENGINE to change engines.')
        super().load_model(model_name)

    def _check_model_cached(self) -> bool:
        model_dir = self.cache_dir / self.MODEL_DIR
        # CosyVoice2 模型目录包含 cosyvoice2.yaml 和多个子模型
        return (model_dir / self.CONFIG_FILE).exists()

    def _auto_download(self) -> None:
        """从 ModelScope 下载 CosyVoice2-0.5B 模型。"""
        from modelscope.hub.snapshot_download import snapshot_download

        model_dir = self.cache_dir / self.MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[cosyvoice2] Auto-downloading from ModelScope: %s", self.MODEL_REPO)

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
                        logger.info("[cosyvoice2] Copied: %s", f.name)
            else:
                for root, dirs, files in os.walk(tmp_dir):
                    for f in files:
                        src_file = Path(root) / f
                        if not (model_dir / f).exists():
                            shutil.copy2(str(src_file), str(model_dir / f))
                            logger.info("[cosyvoice2] Copied: %s", f)

        if not self._check_model_cached():
            raise FileNotFoundError(
                f"Auto-download completed but cosyvoice2.yaml not found in {model_dir}"
            )
        logger.info("[cosyvoice2] Auto-download complete")

    def _load_model(self) -> None:
        """加载 CosyVoice2 模型到 GPU。"""
        import sys

        # CosyVoice 不是标准 pip 包，需要把源码目录加入 PYTHONPATH
        cosyvoice_src = os.getenv(
            "COSYVOICE_SRC_DIR",
            os.path.join(os.path.dirname(__file__), "..", "cosyvoice-src"),
        )
        cosyvoice_src = str(Path(cosyvoice_src).resolve())
        if cosyvoice_src not in sys.path:
            sys.path.insert(0, cosyvoice_src)
        logger.info("[cosyvoice2] PYTHONPATH += %s", cosyvoice_src)

        # Matcha-TTS 子模块也需要加入 PYTHONPATH
        matcha_src = os.path.join(cosyvoice_src, "third_party", "Matcha-TTS")
        if Path(matcha_src).is_dir() and matcha_src not in sys.path:
            sys.path.insert(0, matcha_src)
            logger.info("[cosyvoice2] PYTHONPATH += %s", matcha_src)

        model_dir = str(self.cache_dir / self.MODEL_DIR)
        logger.info("[%s] Loading from %s (device=%s)", self.MODEL_NAME, model_dir, self._device)

        # Hack: huggingface_hub 版本检测在 uv venv 下返回 None，手动注入
        import importlib.metadata
        _orig_version = importlib.metadata.version
        def _patched_version(name):
            if name == "tokenizers":
                return "0.21.4"
            return _orig_version(name)
        importlib.metadata.version = _patched_version

        # wetext's Normalizer() calls snapshot_download("pengzhendong/wetext")
        # on every instantiation, which hits ModelScope's API and gets 403
        # rate-limited on the second call (e.g. after model unload+reload).
        # The files are already cached locally, so patch snapshot_download
        # to return the cached path directly instead of making a network request.
        try:
            import wetext.wetext as _wetext_mod
            _wetext_cache = os.path.expanduser(
                "~/.cache/modelscope/models/pengzhendong--wetext/snapshots/master"
            )
            if os.path.isdir(_wetext_cache) and not hasattr(_wetext_mod, "_patched_snapshot_download"):
                _wetext_mod._patched_snapshot_download = _wetext_mod.snapshot_download
                _wetext_mod.snapshot_download = lambda *a, **kw: _wetext_cache
                logger.info("[cosyvoice2] Patched wetext snapshot_download -> %s", _wetext_cache)
        except Exception as exc:
            logger.debug("[cosyvoice2] wetext patch skipped: %s", exc)

        from cosyvoice.cli import cosyvoice

        model_class = getattr(cosyvoice, self.MODEL_CLASS)
        self._model = model_class(model_dir, fp16=self._device == "cuda")
        self._voice_cache.clear()
        self._install_chunk_reset()
        self._install_token_cancellation()
        self._install_decode_timing()
        self._model_name = self.MODEL_NAME
        logger.info("[%s] Model ready (device=%s)", self.MODEL_NAME, self._device)

    # prompt_text for zero-shot voice cloning — maps prompt WAV to its
    # transcript so inference_zero_shot can use it.  When the voice file
    # matches a known prompt, the transcript is used automatically.
    _PROMPT_TEXTS: dict[str, str] = {}

    def _install_chunk_reset(self):
        """Reset each normalized segment, including segments inside one request.

        The HTTP runtime serializes access to this model instance.
        Drain on close so the upstream generator joins its worker and cleans up.
        """
        core = self._model.model
        initial_hop = core.token_hop_len
        initial_scale = getattr(core, 'stream_scale_factor', 2)
        original = core.tts

        def tts(*args, **kwargs):
            if self._cancel_event is not None and self._cancel_event.is_set():
                return
            core.token_hop_len = initial_hop
            # Short early chunks fill the initial playback buffer sooner,
            # especially after CV3 leading silence has been removed.
            small_second = self.MODEL_NAME == 'cosyvoice3' and os.getenv('TTS_EARLY_SMALL_CHUNKS', '1') == '1'
            core.stream_scale_factor = 1 if small_second else initial_scale
            chunks = original(*args, **kwargs)
            try:
                for index, chunk in enumerate(chunks):
                    if not small_second or index >= 1:
                        core.stream_scale_factor = initial_scale
                    yield chunk
            finally:
                try:
                    for _ in chunks:
                        pass
                finally:
                    chunks.close()
                    core.token_hop_len = initial_hop
                    core.stream_scale_factor = initial_scale

        core.tts = tts

    def _install_decode_timing(self):
        core = self._model.model
        original = core.token2wav
        self._decode_previous = core.__dict__.get('token2wav')
        def decode(*args, **kwargs):
            trace = getattr(self, '_trace', None)
            if trace is None:
                return original(*args, **kwargs)
            with trace.stage('audio_decode'):
                return original(*args, **kwargs)
        core.token2wav = decode

    def _install_token_cancellation(self):
        """Stop producing new speech tokens at the next model iteration.

        Keep the upstream tts generator running its normal join/cache cleanup.
        Requests are serialized by the service runtime.
        """
        llm = self._model.model.llm
        if self._token_cancel_llm is not llm:
            self._token_cancel_wrappers = {}
        for name in ('inference', 'inference_bistream'):
            if not hasattr(llm, name):
                continue
            installed = self._token_cancel_wrappers.get(name)
            if installed and llm.__dict__.get(name) is installed[0]:
                continue
            original = getattr(llm, name)
            def cancellable(*args, _original=original, **kwargs):
                event = self._cancel_event
                source = _original(*args, **kwargs)
                try:
                    for index, token in enumerate(measured_chunks(source, getattr(self, "_trace", None), "speech_tokens")):
                        # A short initial token sequence lets the upstream flow
                        # finalize safely even when cancelled before first audio.
                        if index >= 25 and event is not None and event.is_set():
                            break
                        yield token
                finally:
                    source.close()
            self._token_cancel_wrappers[name] = (cancellable, llm.__dict__.get(name))
            setattr(llm, name, cancellable)
        self._token_cancel_llm = llm

    def _cached_voice(self, prompt_wav, prompt_text):
        path = Path(prompt_wav).resolve()
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size, prompt_text)
        if key in self._voice_cache:
            self._voice_cache.move_to_end(key)
            return self._voice_cache[key], True
        speaker = "http-tts-" + hashlib.sha256(repr(key).encode()).hexdigest()
        normalized = self._model.frontend.text_normalize(prompt_text, split=False)
        self._model.add_zero_shot_spk(normalized, str(path), speaker)
        self._voice_cache[key] = speaker
        while len(self._voice_cache) > 8:
            _, expired = self._voice_cache.popitem(last=False)
            self._model.frontend.spk2info.pop(expired, None)
        return speaker, False

    def unload(self):
        # Break the instance-method closure before releasing GPU allocations.
        if self._model is not None:
            self._model.model.__dict__.pop("tts", None)
            if hasattr(self, "_decode_previous"):
                if self._decode_previous is None:
                    self._model.model.__dict__.pop("token2wav", None)
                else:
                    self._model.model.token2wav = self._decode_previous
                del self._decode_previous
            llm = getattr(self._model.model, 'llm', None)
            if llm is not None:
                for name in ('inference', 'inference_bistream'):
                    installed = self._token_cancel_wrappers.get(name)
                    if installed and llm.__dict__.get(name) is installed[0]:
                        if installed[1] is None:
                            llm.__dict__.pop(name, None)
                        else:
                            setattr(llm, name, installed[1])
        self._voice_cache.clear()
        self._token_cancel_llm = None
        self._token_cancel_wrappers.clear()
        super().unload()

    def _register_prompt_text(self, prompt_wav: str, text: str) -> None:
        """Register a transcript for a prompt WAV file."""
        self._PROMPT_TEXTS[os.path.abspath(prompt_wav)] = text

    def _resolve_prompt(self, voice: Optional[str]) -> tuple:
        """Resolve ``(prompt_wav, prompt_text)`` — single source for both paths.

        ``voice`` is a prompt-audio file path when given and existing; otherwise
        the repo's stock ``zero_shot_prompt.wav``. Shared by the file path and
        the streaming path so the two can never drift apart.
        """
        # 参考音频：voice 参数是文件路径，或默认用仓库自带样本
        cosyvoice_src = os.getenv(
            "COSYVOICE_SRC_DIR",
            os.path.join(os.path.dirname(__file__), "..", "cosyvoice-src"),
        )
        default_prompt = os.path.join(cosyvoice_src, "asset", "zero_shot_prompt.wav")
        prompt_wav = voice if voice and os.path.isfile(voice) else default_prompt
        prompt_text = self._PROMPT_TEXTS.get(os.path.abspath(prompt_wav), "")
        return prompt_wav, prompt_text

    def synthesize_chunks(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        language: Optional[str] = None,
    ):
        """Yield one dict per model audio chunk, as the model streams it.

        Yields ``{"idx": int, "speech": np.float32 1-D mono, "sample_rate": int}``.
        Single source of truth for chunk iteration: ``synthesize()`` consumes
        this generator and the HTTP streaming endpoint consumes it directly, so
        the file path and the real-time path always produce the same audio.

        Semantics mirror the legacy loop 1:1 (same inference-mode selection,
        same CPU/squeeze normalization, same skip rules). Model errors surface
        on the first ``next()`` / iteration — same as the legacy path.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        trace = getattr(self, "_trace", None)
        stage = trace.stage if trace else lambda name: nullcontext()
        prompt_wav, prompt_text = self._resolve_prompt(voice)
        started = time.perf_counter()
        with stage("reference"):
            speaker, cache_hit = self._cached_voice(prompt_wav, prompt_text)
        if trace:
            trace.emit("reference", cache_hit=cache_hit)
        logger.info("timing voice_cache_hit=%s reference_ms=%.2f", cache_hit,
                    (time.perf_counter() - started) * 1000)

        logger.info(
            "[%s] Synthesizing chunks: %d chars, prompt=%s (zero_shot=%s)",
            self.MODEL_NAME, len(text), prompt_wav, bool(prompt_text),
        )

        chunks = self._inference_chunks(text, prompt_text, prompt_wav, speaker)
        from ..audio_processing import LeadingSilenceTrimmer
        sr = int(self._model.sample_rate)
        trimmer = (LeadingSilenceTrimmer(sr) if self.MODEL_NAME == 'cosyvoice3'
                   and os.getenv('TTS_TRIM_LEADING_SILENCE', '1') == '1' else None)

        try:
            idx = 0
            raw_idx = 0
            for chunk in measured_chunks(chunks, trace):
                if not isinstance(chunk, dict):
                    continue
                speech = chunk.get("tts_speech")
                if speech is None:
                    continue
                if hasattr(speech, "cpu"):
                    with stage("to_cpu"):
                        speech = speech.cpu().numpy()
                if speech.ndim > 1:
                    speech = speech.squeeze()
                sr = int(chunk.get("sample_rate", self._model.sample_rate))
                raw_idx += 1
                raw_samples = len(speech)
                envelope = onset_metrics(speech, sr) if trace and raw_idx <= 3 else {}
                if trimmer is not None:
                    with stage("leading_trim"):
                        speech = trimmer.feed(speech)
                    if trace and raw_idx <= 3:
                        trace.emit("onset", **envelope, raw_index=raw_idx, raw_audio_ms=round(raw_samples / sr * 1000, 2),
                                   output_audio_ms=round(len(speech) / sr * 1000, 2),
                                   waiting_for_onset=not trimmer.open,
                                   trimmed_ms=round(trimmer.trimmed / sr * 1000, 2))
                    if not len(speech):
                        continue
                yield {"idx": idx, "speech": speech, "sample_rate": sr}
                idx += 1
            if trimmer is not None:
                tail = trimmer.flush()
                if len(tail):
                    yield {"idx": idx, "speech": tail, "sample_rate": sr}
                logger.info("timing leading_trim_ms=%.2f", trimmer.trimmed / sr * 1000)

        finally:
            for _ in chunks:
                pass
            chunks.close()

    def _inference_chunks(self, text, prompt_text, prompt_wav, speaker):
        if prompt_text:
            # Use inference_zero_shot — better quality, needs prompt_text
            logger.info("[%s] Using zero_shot mode (prompt_text=%d chars)", self.MODEL_NAME, len(prompt_text))
            chunks = self._model.inference_zero_shot(text, prompt_text, prompt_wav, zero_shot_spk_id=speaker, stream=True)
        else:
            # Fallback: inference_cross_lingual — no prompt_text needed
            logger.info("[%s] Using cross_lingual mode (no prompt_text)", self.MODEL_NAME)
            chunks = self._model.inference_cross_lingual(text, prompt_wav, zero_shot_spk_id=speaker, stream=True)

        return chunks

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        language: Optional[str] = None,
    ) -> SynthesizeResult:
        """合成语音，写入 output_path。

        CosyVoice2-0.5B 不支持预设音色（SFT），使用零样本声音克隆。
        优先使用 inference_zero_shot（需要 prompt_text），效果更好；
        如果没有 prompt_text 则回退到 inference_cross_lingual。

        - voice: 参考音频文件路径（WAV），默认用仓库自带的 zero_shot_prompt.wav
        - speed: 1.0 = 正常速度（CosyVoice2 不直接支持，忽略）
        - language: 语言提示（CosyVoice2 自动处理，忽略）

        流式推理本来就是逐块产出；本方法全量收齐 chunk 后写一个文件，
        行为与旧实现一致（同一套 chunk 语义，见 synthesize_chunks）。
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        import numpy as np
        import soundfile as sf

        start = time.time()

        # 收集所有音频 chunk
        audio_chunks = []
        sample_rate = 24000  # CosyVoice2 默认采样率
        for item in self.synthesize_chunks(
            text, voice=voice, speed=speed, language=language
        ):
            audio_chunks.append(item["speech"])
            sample_rate = item["sample_rate"]

        processing_ms = int((time.time() - start) * 1000)

        if not audio_chunks:
            return SynthesizeResult(
                audio_path="",
                processing_time_ms=processing_ms,
                error="No audio generated",
            )

        # 拼接所有 chunk
        audio = np.concatenate(audio_chunks)

        # 写入文件
        sf.write(output_path, audio, sample_rate)

        duration_ms = int(len(audio) / sample_rate * 1000)

        logger.info(
            "[cosyvoice2] Synthesis complete: %d chars -> %.1fs audio (processing: %dms)",
            len(text), duration_ms / 1000, processing_ms,
        )

        return SynthesizeResult(
            audio_path=output_path,
            processing_time_ms=processing_ms,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
        )

