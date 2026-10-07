"""Materialize a self-contained speech runtime and model assets on Windows.

No source directories are removed. Destination changes stay within --root.
TTS libraries retain precedence; STT-only distributions are copied to an internal
supplement directory so ordinary package and namespace-package lookup is preserved.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import venv
import threading
from concurrent.futures import ThreadPoolExecutor

def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()

def inside(root, path):
    resolved = Path(path).resolve()
    resolved.relative_to(root)
    if resolved == root:
        raise ValueError("Refuse operation on the destination root itself")
    return resolved

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

class Copier:
    def __init__(self, root):
        self.root = root
        self.files = 0
        self.bytes = 0
        self.last_report = time.monotonic()
        self.records = []
        self.lock = threading.Lock()

    def file(self, source, target):
        target = inside(self.root, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size == source.stat().st_size:
            original_hash = digest(source)
            if digest(target) == original_hash:
                self.record(source, target, original_hash)
                return
        h = hashlib.sha256()
        with source.open("rb") as inp, target.open("wb") as out:
            while chunk := inp.read(4*1024*1024):
                out.write(chunk)
                h.update(chunk)
        if digest(target) != h.hexdigest():
            raise RuntimeError("Copied file checksum mismatch: " + str(target))
        shutil.copystat(source, target)
        self.record(source, target, h.hexdigest())

    def record(self, source, target, checksum):
        size = target.stat().st_size
        with self.lock:
            self.records.append({"source": str(source), "staged": str(target),
                                 "bytes": size, "sha256": checksum})
            self.files += 1
            self.bytes += size
            if time.monotonic()-self.last_report > 15:
                print(json.dumps({"copied_files": self.files, "copied_gb": round(self.bytes/1e9, 2)}), flush=True)
                self.last_report = time.monotonic()

    def tree(self, source, target):
        jobs = []
        for parent, dirs, names in os.walk(source):
            dirs[:] = [n for n in dirs if n not in {".git", "__pycache__"}]
            relative = Path(parent).relative_to(source)
            inside(self.root, target/relative).mkdir(parents=True, exist_ok=True)
            for name in names:
                if name == ".git" or name.endswith((".pyc", ".pyo")):
                    continue
                jobs.append((Path(parent)/name, target/relative/name))
        with ThreadPoolExecutor(max_workers=8) as pool:
            for _ in pool.map(lambda pair: self.file(*pair), jobs):
                pass

def distributions(path):
    return {canonical(d.metadata["Name"]): d for d in metadata.distributions(path=[str(path)])
            if d.metadata.get("Name")}

def rewrite_runtime_paths(environment, replacements, final_env):
    changed = []
    candidates = list((environment/"Lib").rglob("*.pth"))
    candidates += list((environment/"Lib").rglob("*.egg-link"))
    candidates += list((environment/"Lib").rglob("__editable__*.py"))
    candidates += list((environment/"Lib").rglob("direct_url.json"))
    candidates += [p for p in (environment/"Scripts").iterdir()
                   if p.is_file() and p.suffix.lower() in ("", ".ps1", ".bat", ".py")]
    candidates += [environment/"pyvenv.cfg"]
    replacements = {**replacements, str(environment): str(final_env)}
    for path in candidates:
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        content = original
        for source, target in replacements.items():
            for old, new in ((source.replace("\\","/"), target.replace("\\","/")),
                             (source.replace("\\","\\\\"), target.replace("\\","\\\\")),
                             (source, target)):
                content = content.replace(old, new)
        if content != original:
            path.write_text(content, encoding="utf-8")
            changed.append(str(path))
    return changed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tts", type=Path, required=True)
    parser.add_argument("--stt", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    root, tts, stt = [p.resolve() for p in (args.root, args.tts, args.stt)]
    if root in (tts, stt) or root.is_relative_to(tts) or root.is_relative_to(stt):
        raise ValueError("Destination must be separate from old service roots")
    for parent in (tts, stt):
        for name in ("models", "venv/Lib/site-packages"):
            if not (parent/name).is_dir():
                raise FileNotFoundError(parent/name)
    tts_lib, stt_lib = [p/"venv/Lib/site-packages" for p in (tts, stt)]
    primary, secondary = distributions(tts_lib), distributions(stt_lib)
    extras = {name: d for name, d in secondary.items() if name not in primary}
    selected = set()
    for dist in extras.values():
        for item in dist.files or ():
            candidate = (stt_lib/item).resolve()
            if candidate.is_relative_to(stt_lib) and candidate.is_file() and candidate.suffix not in (".pyc", ".pyo"):
                selected.add(candidate)
    summary = {"primary_distributions": len(primary), "stt_only_distributions": len(extras),
               "stt_only_names": sorted(extras), "supplement_bytes": sum(p.stat().st_size for p in selected),
               "destination": str(root), "source_directories_preserved": True}
    print(json.dumps(summary), flush=True)
    if not args.apply:
        return
    for target in ("models/stt", "models/tts", "cosyvoice-src"):
        if (root/target).exists():
            raise FileExistsError("Destination asset directory already exists: " + target)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = inside(root, args.resume or root/"runtime"/("asset-migration-"+stamp))
    work.relative_to(root/"runtime")
    if not work.name.startswith("asset-migration-") or (work/"result.json").exists():
        raise ValueError("Resume must refer to an unfinished asset-migration staging directory")
    stage = inside(root, work/"stage")
    stage.mkdir(parents=True, exist_ok=bool(args.resume))
    env = stage/"venv"
    # Run using the base Python so pyvenv.cfg points to the stable base interpreter.
    if not (env/"Scripts/python.exe").is_file():
        venv.EnvBuilder(with_pip=False).create(env)
    copier = Copier(root)
    for source, destination in ((tts_lib, env/"Lib/site-packages"),
                                (stt/"models", stage/"models/stt"),
                                (tts/"models", stage/"models/tts"),
                                (tts/"cosyvoice-src", stage/"cosyvoice-src")):
        print(json.dumps({"copying": str(source), "to": str(destination)}), flush=True)
        copier.tree(source, destination)
    supplement = env/"Lib/speech-stt-site-packages"
    supplement.mkdir(parents=True)
    for source in sorted(selected):
        copier.file(source, supplement/source.relative_to(stt_lib))
    (env/"Lib/site-packages/speech-stt-supplement.pth").write_text("../speech-stt-site-packages\n", encoding="utf-8")
    replacements = {str(tts/"cosyvoice-src"): str(root/"cosyvoice-src"),
                    str(tts/"models"): str(root/"models/tts"),
                    str(stt/"models"): str(root/"models/stt"),
                    str(tts/"venv"): str(root/"venv"),
                    str(stt/"venv"): str(root/"venv")}
    rewrites = rewrite_runtime_paths(env, replacements, root/"venv")
    # Rebuild console launchers; copied Windows launchers embed obsolete interpreters.
    sys.path.insert(0, str(env/"Lib/site-packages"))
    from distlib.scripts import ScriptMaker
    maker = ScriptMaker(None, str(env/"Scripts"))
    maker.executable = str(root/"venv/Scripts/python.exe")
    maker.clobber = True
    maker.variants = {""}
    scripts = []
    for dist in list(primary.values()) + list(extras.values()):
        for entry in dist.entry_points:
            if entry.group in ("console_scripts", "gui_scripts"):
                scripts.extend(maker.make(entry.name+" = "+entry.value, options={"gui": entry.group == "gui_scripts"}))
    config_path = inside(root, root/"service.local.json")
    original = config_path.read_bytes()
    config = json.loads(original)
    cfg = config["environment"]
    cfg["STT_MODEL_DIR"] = str(root/"models/stt")
    cfg["TTS_MODEL_DIR"] = str(root/"models/tts")
    cfg["COSYVOICE_SRC_DIR"] = str(root/"cosyvoice-src")
    voice = cfg.get("TTS_DEFAULT_VOICE")
    if voice and Path(voice).resolve().is_relative_to(tts/"models"):
        cfg["TTS_DEFAULT_VOICE"] = str(root/"models/tts"/Path(voice).resolve().relative_to(tts/"models"))
    (work/"service.before.json").write_bytes(original)
    library_path = inside(root, root/"runtime-libraries.json")
    if library_path.exists():
        (work/"runtime-libraries.before.json").write_bytes(library_path.read_bytes())
    # Validate all move paths before the first rename; no recursive deletion is used.
    pairs = [(inside(root, stage/name), inside(root, root/name))
             for name in ("models/stt", "models/tts", "cosyvoice-src", "venv")]
    old_env = inside(root, root/"venv")
    saved_env = inside(root, work/"previous-shared-venv")
    moved = []
    try:
        if old_env.exists():
            old_env.rename(saved_env)
        for source, target in pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            moved.append((source, target))
        config_path.write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
        libraries = {"kind": "independent-local-snapshot", "python_base": sys.base_prefix,
                     "libraries": [str(root/"venv/Lib/site-packages"), str(root/"venv/Lib/speech-stt-site-packages")],
                     "primary_versions": {k:d.version for k,d in primary.items()},
                     "stt_supplement_versions": {k:d.version for k,d in extras.items()}}
        library_path.write_text(json.dumps(libraries, indent=2)+"\n", encoding="utf-8")
    except BaseException:
        for source, target in reversed(moved):
            source.parent.mkdir(parents=True, exist_ok=True)
            target.rename(source)
        if saved_env.exists():
            saved_env.rename(old_env)
        config_path.write_bytes(original)
        if (work/"runtime-libraries.before.json").exists():
            library_path.write_bytes((work/"runtime-libraries.before.json").read_bytes())
        raise
    summary.update(status="materialized", copied_files=copier.files, copied_bytes=copier.bytes,
                   path_rewrites=len(rewrites), launchers_rebuilt=len(scripts), rollback_directory=str(work))
    (work/"copied-files.json").write_text(json.dumps(copier.records, indent=2), encoding="utf-8")
    (work/"result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)

if __name__ == "__main__":
    main()
