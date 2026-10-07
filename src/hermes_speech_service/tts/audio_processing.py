"""Conservative leading-only silence removal for locally generated audio."""
import numpy as np


class LeadingSilenceTrimmer:
    def __init__(self, sample_rate, threshold=.001, preroll_ms=100, max_wait_ms=3000):
        self.frame = max(1, int(sample_rate * .02))
        self.preroll = int(sample_rate * preroll_ms / 1000)
        self.limit = int(sample_rate * max_wait_ms / 1000)
        self.threshold = threshold
        self.buffer = np.empty(0, dtype=np.float32)
        self.open = False
        self.trimmed = 0

    def feed(self, audio):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if self.open:
            return audio
        self.buffer = np.concatenate((self.buffer, audio))
        frames = len(self.buffer) // self.frame
        if frames:
            rms = np.sqrt(np.mean(self.buffer[:frames * self.frame].reshape(-1, self.frame) ** 2, axis=1))
            active = np.flatnonzero(rms >= self.threshold)
            if len(active):
                self.trimmed = max(0, int(active[0]) * self.frame - self.preroll)
                return self._release()
        if len(self.buffer) >= self.limit:
            # Bound buffering; preserve quiet audio beyond the observation window.
            self.trimmed = max(0, self.limit - self.preroll)
            return self._release()
        return np.empty(0, dtype=np.float32)

    def _release(self):
        result = self.buffer[self.trimmed:]
        self.buffer = np.empty(0, dtype=np.float32)
        self.open = True
        return result

    def flush(self):
        # Entirely quiet short output is preserved for diagnosis.
        return self._release() if not self.open else np.empty(0, dtype=np.float32)

