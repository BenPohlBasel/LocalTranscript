"""Pfade + Werkzeuge (whisper-cli, ffmpeg, Modelle) + App-Einstellungen.

Zwei Betriebsarten:
- dev: Repo-Checkout — Werkzeuge aus Homebrew/PATH, Modelle aus
  LT_MODELS_DIR | <repo>/models | ~/whisper-models.
- bundle (LT_BUNDLED=1 oder Marker-Datei BUNDLED im App-Root): alles
  aus den mitgelieferten Resources (bin/, lib/, models/, venv/).

Einstellungen (Bibliotheks-Wurzel, Defaults) leben als JSON unter
~/Library/Application Support/LocalTranscript/config.json — das BACKEND
besitzt die Config (v1: Electron-main.js besaß sie; die Shell soll
dumm sein).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

APP_NAME = "LocalTranscript"
APP_VERSION = "2.0.0"
#: DER LocalTranscript-Port (2026-09-09): 5628 = „LOCT" auf der
#: Telefontastatur — enrich 36742 = „ENRIC", Zotero-Tradition
#: (23119 = „ZOT"). Vier Buchstaben, nicht fünf: „LOCTR" wäre 56287
#: und läge im EPHEMEREN Bereich, den macOS selbst verteilt
#: (49152–65535) — als fester Dienst-Port untauglich. 5628 liegt im
#: User-Bereich 1024–49151 und ist IANA-unvergeben.
#: Override: LT_SERVE_PORT (Shell und Frontend kennen ihn auch).
PORT = int(os.environ.get("LT_SERVE_PORT") or 5628)


def get_app_root() -> Path:
    """Bundle: Resources-Ordner; dev: Repo-Wurzel (backend/..)."""
    env = os.environ.get("LT_APP_ROOT")
    if env:
        return Path(env)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent.parent / "Resources"
    return Path(__file__).resolve().parent.parent.parent.parent


def is_bundled() -> bool:
    return (os.environ.get("LT_BUNDLED") == "1"
            or (get_app_root() / "BUNDLED").exists())


def _find_executable(name: str, bundled: Path) -> str:
    kandidaten: list[Path] = []
    if is_bundled():
        kandidaten.append(bundled)
    kandidaten += [Path("/opt/homebrew/bin") / name,
                   Path("/usr/local/bin") / name]
    if not is_bundled():
        kandidaten.append(bundled)
    for k in kandidaten:
        if k.is_file() and os.access(k, os.X_OK):
            return str(k)
    w = shutil.which(name)
    if w:
        return w
    raise FileNotFoundError(
        f"{name} nicht gefunden — brew install "
        f"{'whisper-cpp' if 'whisper' in name else name}")


def get_whisper_cli() -> str:
    return _find_executable("whisper-cli", get_app_root() / "bin" / "whisper-cli")


def get_ffmpeg_cli() -> str:
    return _find_executable("ffmpeg", get_app_root() / "bin" / "ffmpeg")


def get_models_dir() -> Path:
    env = os.environ.get("LT_MODELS_DIR")
    if env:
        return Path(env)
    app_models = get_app_root() / "models"
    if any(app_models.glob("ggml-*.bin")) if app_models.is_dir() else False:
        return app_models
    home_models = Path.home() / "whisper-models"
    if home_models.is_dir() and any(home_models.glob("ggml-*.bin")):
        return home_models
    return app_models


def get_available_models() -> list[dict]:
    d = get_models_dir()
    aus = []
    if d.is_dir():
        for p in sorted(d.glob("ggml-*.bin")):
            aus.append({"name": p.stem.replace("ggml-", ""),
                        "size_mb": round(p.stat().st_size / 1e6, 1)})
    aus.sort(key=lambda m: m["size_mb"])
    return aus


# ---------- Einstellungen ----------

def _config_dir() -> Path:
    env = os.environ.get("LT_CONFIG_DIR")
    if env:
        return Path(env)
    return (Path.home() / "Library" / "Application Support" / APP_NAME)


def _config_file() -> Path:
    return _config_dir() / "config.json"


DEFAULTS = {
    "library_root": "",       # "" = noch nicht gewählt (First-Run)
    "model": "large-v3-turbo",
    "language": "de",
    "diarize": True,
    "speaker_range": "auto",  # "auto" | "min-max"
    "cluster_threshold": 0.5,
    "ui_language": "de",
}


def read_config() -> dict:
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(_config_file().read_text("utf-8")))
    except (OSError, ValueError):
        pass
    # verwaister Speicherort ⇒ wie ungesetzt (First-Run erscheint wieder)
    root = cfg.get("library_root") or ""
    if root and not Path(root).is_dir():
        cfg["library_root"] = ""
    return cfg


def write_config(aenderungen: dict) -> dict:
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(_config_file().read_text("utf-8")))
    except (OSError, ValueError):
        pass
    cfg.update({k: v for k, v in aenderungen.items() if k in DEFAULTS})
    _config_dir().mkdir(parents=True, exist_ok=True)
    tmp = _config_file().with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), "utf-8")
    tmp.replace(_config_file())
    return read_config()


def default_library_root() -> Path:
    return Path.home() / "Documents" / APP_NAME


def library_root() -> Path | None:
    root = read_config().get("library_root") or ""
    return Path(root) if root else None
