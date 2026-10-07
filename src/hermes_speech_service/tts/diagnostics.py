"""Bounded, request-scoped TTS diagnostics; no text or audio payloads."""
from contextlib import contextmanager
import json
import logging
import subprocess
import threading
import time

log = logging.getLogger("tts-service")

class RequestTrace:
    def __init__(self, request_id):
        self.request_id = request_id
        self.started = time.perf_counter()
        self.stages = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None

    def emit(self, kind, **values):
        log.info("tts_diagnostic %s", json.dumps(dict(
            request_id=self.request_id, kind=kind,
            elapsed_ms=round((time.perf_counter()-self.started)*1000, 2),
            **values), separators=(",", ":")))

    @contextmanager
    def stage(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter()-started)*1000
            with self.lock:
                s = self.stages.setdefault(name, dict(count=0, total_ms=0, max_ms=0, first_ms=ms))
                s["count"] += 1
                s["total_ms"] += ms
                s["max_ms"] = max(s["max_ms"], ms)

    def checkpoint(self, name, **values):
        with self.lock:
            stages = {k: {n: round(v, 2) for n, v in s.items()} for k, s in self.stages.items()}
        self.emit(name, stages=stages, **values)

    def start_resources(self):
        self.thread = threading.Thread(target=self._sample, daemon=True, name="tts-resources")
        self.thread.start()

    def _sample(self):
        try:
            import psutil
            process = psutil.Process()
            process.cpu_percent()
        except Exception:
            process = None
        while not self.stop.is_set():
            sample = {}
            if process is not None:
                try:
                    sample.update(process_cpu_percent=process.cpu_percent(),
                                  rss_mb=round(process.memory_info().rss/1048576, 2),
                                  system_cpu_percent=psutil.cpu_percent(),
                                  available_memory_mb=round(psutil.virtual_memory().available/1048576, 2))
                except Exception:
                    sample["cpu_memory_unavailable"] = True
            else:
                sample["cpu_memory_unavailable"] = True
            try:
                result = subprocess.run([
                    "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,pstate,clocks.sm",
                    "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if result.returncode == 0:
                    sample["gpu"] = [row.strip().split(", ") for row in result.stdout.splitlines()][:8]
                else:
                    sample["gpu_unavailable"] = True
            except (OSError, subprocess.TimeoutExpired):
                sample["gpu_unavailable"] = True
            if self.stop.is_set():
                break
            self.emit("resources", **sample)
            self.stop.wait(2)

    def close(self, cancelled=False):
        self.stop.set()
        # Never wait for the resource subprocess on the audio producer thread.
        self.checkpoint("finished", cancelled=cancelled)


def measured_chunks(chunks, trace, name="model_next"):
    iterator = iter(chunks)
    while True:
        try:
            if trace is None:
                item = next(iterator)
            else:
                with trace.stage(name):
                    item = next(iterator)
        except StopIteration:
            return
        yield item


def onset_metrics(speech, sample_rate):
    """Numeric envelope only, for the first few raw chunks."""
    import numpy as np
    audio = np.asarray(speech).reshape(-1)
    frame = max(1, int(sample_rate * .02))
    usable = len(audio) // frame * frame
    if not usable:
        return {"max_frame_rms": None, "first_active_ms": {}}
    rms = np.sqrt(np.mean(audio[:usable].reshape(-1, frame) ** 2, axis=1))
    positions = {}
    for threshold in (.0001, .0005, .001):
        active = np.flatnonzero(rms >= threshold)
        positions[str(threshold)] = round(int(active[0]) * frame / sample_rate * 1000, 2) if len(active) else None
    return {"max_frame_rms": round(float(rms.max()), 7), "first_active_ms": positions}

