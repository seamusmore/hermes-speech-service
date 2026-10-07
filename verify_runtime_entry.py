"""Validation-only launcher: reject Python access to both retired service roots."""
import atexit
import json
import os
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
blocked = [os.path.normcase(os.path.abspath(path)) for path in
           json.loads(os.environ["SPEECH_VALIDATION_BLOCKED_ROOTS"])]
attempts = []

def old_path(value):
    if not isinstance(value, (str, bytes, os.PathLike)):
        return False
    value = os.path.normcase(os.path.abspath(os.fsdecode(value)))
    return any(value == prefix or value.startswith(prefix+os.sep) for prefix in blocked)

def audit(event, args):
    if event in ("open", "os.listdir", "os.scandir", "os.chdir", "os.add_dll_directory", "ctypes.dlopen"):
        if args and old_path(args[0]):
            attempts.append({"event": event, "path": os.fsdecode(args[0])})
            raise PermissionError("Validation blocked access to a retired service directory")

sys.addaudithook(audit)
# -S lets the guard run before .pth files. Restore this venv's normal site setup.
sys.prefix = sys.exec_prefix = str(root/"venv")
import site
site.addsitedir(str(root/"venv/Lib/site-packages"))
sys.path.insert(0, str(root))

def report():
    names = ("torch", "numpy", "transformers", "funasr", "cosyvoice", "matcha", "onnxruntime")
    origins = {name: getattr(sys.modules.get(name), "__file__", None) for name in names}
    native_old_paths = []
    try:
        import psutil
        native_old_paths = [item.path for item in psutil.Process().memory_maps() if old_path(item.path)]
    except Exception:
        native_old_paths = ["native_mapping_inspection_unavailable"]
    output = {"blocked_roots": blocked, "attempted_old_access": attempts,
              "module_origins": origins, "native_old_paths": native_old_paths}
    (root/"runtime/independent-runtime-access.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

atexit.register(report)
import uvicorn
from uvicorn.importer import import_from_string

def guarded_run(application, **options):
    factory = options.pop("factory", False)
    if isinstance(application, str):
        application = import_from_string(application)
    if factory:
        application = application()
    # Surface model-load diagnostics in this validation log while production
    # HTTP errors remain sanitized.
    import importlib
    import traceback
    for name in ("stt", "tts"):
        component = importlib.import_module("hermes_speech_service."+name+".app")
        original = component.warmup
        def traced_warmup(original=original):
            try:
                return original()
            except Exception:
                traceback.print_exc()
                raise
        component.warmup = traced_warmup
    server = uvicorn.Server(uvicorn.Config(application, **options))
    @application.post("/__validation_shutdown")
    async def stop_validation():
        report()
        server.should_exit = True
        return {"stopping": True}
    server.run()

uvicorn.run = guarded_run
# Graceful shutdown executes the audit report before this process exits.
sys.argv = [str(root/"run.py"), *sys.argv[1:]]
runpy.run_path(str(root/"run.py"), run_name="__main__")
