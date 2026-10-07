"""Compatibility entry for physical model and dependency migration (no downloads)."""
import argparse
from pathlib import Path
import subprocess
import sys


def prepare(root, libraries):
    root = Path(root).resolve()
    if len(libraries) != 2:
        raise ValueError("Expected TTS and STT environment paths")
    tts, stt = [Path(item).resolve().parent for item in libraries]
    subprocess.run([getattr(sys, "_base_executable", sys.executable), "-I", "-B",
                    str(root/"migrate_assets.py"), "--root", str(root),
                    "--tts", str(tts), "--stt", str(stt), "--apply"], check=True)
    return root / "venv" / "Scripts" / "python.exe"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts-runtime", required=True)
    parser.add_argument("--stt-runtime", required=True)
    args = parser.parse_args()
    print(prepare(Path(__file__).resolve().parent, [args.tts_runtime, args.stt_runtime]))
