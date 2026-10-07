"""Real unified HTTP roundtrip with installed weights; no speaker playback."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import requests

root = Path(__file__).resolve().parent
output = root/"runtime"
output.mkdir(exist_ok=True)
with socket.socket() as reserve:
    reserve.bind(("127.0.0.1", 0))
    port = reserve.getsockname()[1]
base = "http://127.0.0.1:" + str(port)
results = {"port": port, "physical_playback": "not_measured"}
with (output/"probe-service.log").open("wb") as log:
    command = [sys.executable, "-B", str(root/"run.py"), "--port", str(port)]
    if os.getenv("SPEECH_VALIDATION_BLOCKED_ROOTS"):
        command = [sys.executable, "-I", "-S", "-B", str(root/"verify_runtime_entry.py"), "--port", str(port)]
    child = subprocess.Popen(command,
        stdout=log, stderr=log, stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        for _ in range(600):
            if child.poll() is not None:
                raise RuntimeError("Service exited; inspect probe-service.log")
            try:
                ready = requests.get(base+"/health", timeout=1)
                ready.raise_for_status()
                if ready.json()["service"] == "hermes-speech-service":
                    break
            except requests.RequestException:
                time.sleep(.2)
        else:
            raise TimeoutError("Service startup timeout")
        print("service HTTP ready", flush=True)
        started = time.monotonic()
        warm = requests.post(base+"/warmup", timeout=420)
        warm.raise_for_status()
        results["warmup"] = warm.json()
        results["warmup_seconds"] = round(time.monotonic()-started, 2)
        print(json.dumps({"warmup": results["warmup"]}, ensure_ascii=True), flush=True)
        if not warm.json()["ready"]:
            raise RuntimeError("Model warmup failed; inspect probe-service.log")
        audio = requests.post(base+"/tts/synthesize", data={"text": "这是统一语音服务的测试。今天的天气很好。我们继续下一轮。"}, timeout=180)
        audio.raise_for_status()
        (output/"probe.wav").write_bytes(audio.content)
        results["audio_bytes"] = len(audio.content)
        with (output/"probe.wav").open("rb") as handle:
            transcript = requests.post(base+"/stt/transcribe", files={"file": ("probe.wav", handle, "audio/wav")}, timeout=120)
        transcript.raise_for_status()
        results["transcription"] = transcript.json()
        results["ready"] = requests.get(base+"/ready", timeout=5).json()
        (output/"probe-result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(results, ensure_ascii=True), flush=True)
    finally:
        # The guarded process uses an HTTP shutdown callback supplied below.
        if os.getenv("SPEECH_VALIDATION_BLOCKED_ROOTS"):
            try:
                requests.post(base+"/__validation_shutdown", timeout=2)
            except requests.RequestException:
                pass
        else:
            child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
