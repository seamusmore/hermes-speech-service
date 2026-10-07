"""Serialize model use with idle unloading; measure time with a monotonic clock."""
import threading
import time
from contextlib import contextmanager


class ModelRuntime:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_activity = time.monotonic()
        self.active = False
        self.warmed = False

    @property
    def idle_seconds(self):
        return 0.0 if self.active else time.monotonic() - self.last_activity

    @contextmanager
    def request(self):
        started = time.perf_counter()
        with self.lock:
            self.active = True
            timings = {"queue_ms": round((time.perf_counter() - started) * 1000, 2)}
            try:
                yield timings
            finally:
                self.last_activity = time.monotonic()
                self.active = False

    def unload_if_idle(self, engine, timeout):
        if timeout <= 0 or not self.lock.acquire(blocking=False):
            return False
        try:
            if engine.model_loaded and self.idle_seconds >= timeout:
                engine.unload()
                self.warmed = False
                return True
            return False
        finally:
            self.lock.release()

