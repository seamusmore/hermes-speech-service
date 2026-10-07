"""CosyVoice3 adapter using the existing HTTP, cache and lifecycle contracts."""
from . import EngineInfo, register_engine
from .cosyvoice2 import CosyVoice2Engine


@register_engine('cosyvoice3')
class CosyVoice3Engine(CosyVoice2Engine):
    MODEL_NAME = 'cosyvoice3'
    MODEL_CLASS = 'CosyVoice3'
    CONFIG_FILE = 'cosyvoice3.yaml'
    MODEL_DIR = 'Fun-CosyVoice3-0.5B-2512'
    MODEL_REPO = 'FunAudioLLM/Fun-CosyVoice3-0.5B-2512'
    PROMPT_PREFIX = 'You are a helpful assistant.<|endofprompt|>'
    REQUIRED_FILES = ('cosyvoice3.yaml', 'llm.pt', 'flow.pt', 'hift.pt',
                      'campplus.onnx', 'speech_tokenizer_v3.onnx',
                      'CosyVoice-BlankEN/config.json',
                      'CosyVoice-BlankEN/model.safetensors',
                      'CosyVoice-BlankEN/tokenizer_config.json',
                      'CosyVoice-BlankEN/vocab.json', 'CosyVoice-BlankEN/merges.txt')

    def info(self):
        return EngineInfo(name=self.MODEL_NAME, display_name='Fun-CosyVoice3-0.5B-2512',
                          models=[self.MODEL_NAME], default_model=self.MODEL_NAME,
                          supports_streaming=True, supports_voice_cloning=True,
                          voices=[])

    def _check_model_cached(self):
        root = self.cache_dir / self.MODEL_DIR
        return all((root / name).is_file() and (root / name).stat().st_size > 0
                   for name in self.REQUIRED_FILES)

    def _auto_download(self):
        # Installation is explicit so a service request never starts a multi-GB download.
        raise FileNotFoundError(
            f'CosyVoice3 weights incomplete in {self.cache_dir / self.MODEL_DIR}. '
            'Run download_cosyvoice3.py before starting this engine.')

    def _resolve_prompt(self, voice):
        path, transcript = super()._resolve_prompt(voice)
        if transcript and '<|endofprompt|>' not in transcript:
            transcript = self.PROMPT_PREFIX + transcript
        return path, transcript

    def _inference_chunks(self, text, prompt_text, prompt_wav, speaker):
        # CV3 places its system prefix in the prompt for zero-shot, and in the
        # synthesis text for cross-lingual mode (official example.py contract).
        if not prompt_text and '<|endofprompt|>' not in text:
            text = self.PROMPT_PREFIX + text
        return super()._inference_chunks(text, prompt_text, prompt_wav, speaker)

