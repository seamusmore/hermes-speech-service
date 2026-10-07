import argparse
import os
import json
from pathlib import Path
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Unified local STT/TTS service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", type=Path, help="JSON file containing service environment defaults")
    args = parser.parse_args()
    config_path = args.config or Path(__file__).resolve().parents[2] / "service.local.json"
    if config_path.is_file():
        defaults = json.loads(config_path.read_text(encoding="utf-8")).get("environment", {})
        for key, value in defaults.items():
            if not key.startswith(("STT_", "TTS_", "SPEECH_", "COSYVOICE_")):
                parser.error("Unsupported service environment setting: " + key)
            os.environ.setdefault(key, str(value))
    ffmpeg_dir = os.getenv("SPEECH_FFMPEG_DIR")
    if ffmpeg_dir:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    if args.host not in ("127.0.0.1", "::1", "localhost") and not os.getenv("HERMES_SPEECH_SERVICE_TOKEN"):
        parser.error("A service token is required to listen on a non-loopback interface")
    uvicorn.run("hermes_speech_service.app:create_app", factory=True,
                host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
