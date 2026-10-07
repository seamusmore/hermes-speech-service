"""Source-checkout launcher; installed distributions use hermes-speech-service."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from hermes_speech_service.__main__ import main
if __name__ == "__main__":
    main()

