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
import time
from pathlib import Path

APP_NAME = "LocalTranscript"
APP_VERSION = "2.4.0"
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


#: whisper.cpp-Modelldateien beginnen mit dem ggml-Magic 0x67676d6c
#: (little-endian auf der Platte: «lmgg»). Daran erkennt der Scan eine
#: echte Modelldatei — eine umbenannte PDF landet nicht in der Auswahl.
GGML_MAGIC = b"lmgg"
#: Kleiner ist kein Whisper-Modell (tiny = 75 MB): eine abgebrochene
#: Umwandlung hinterlässt Kopf + Tokentabelle (~600 KB) mit gültigem Magic.
MODELL_MIN_BYTES = 30_000_000
#: Eine Datei, die vor weniger als so vielen Sekunden geändert wurde,
#: wird gerade noch kopiert (3 GB aus dem Finder) — noch nicht anbieten.
MODELL_RUHE_S = 5
#: Ordnername in der Bibliothek (User 2026-09-11: «still ein Modell-Ordner
#: im Bibliotheks-Ordner, in den man Modelle dropt»).
EIGENE_MODELLE = "Modelle"


def get_eigene_modelle_dir() -> Path | None:
    """`<Bibliothek>/Modelle/` — der Ordner, in den die Person eigene
    `ggml-*.bin` legt. None, solange keine Bibliothek gewählt ist.
    Nie der Bundle-Ordner: der ist signiert, jede fremde Datei darin
    bricht die Signatur, und jedes Update ersetzt ihn."""
    root = library_root()
    return root / EIGENE_MODELLE if root else None


#: macOS-Dateiflag SF_DATALESS: der Inhalt liegt nur in iCloud («Mac-
#: Speicher optimieren» hat die Datei ausgelagert). Ein open() blockiert
#: dann, bis iCloud die Datei geholt hat — beim 1,6-GB-Modell 16 s, bei
#: schlechtem Netz Minuten; /api/models hing damit (Live-Befund
#: 2026-09-11, Bibliothek in ~/Documents mit iCloud Drive).
SF_DATALESS = 0x40000000


def _dataless(st: os.stat_result) -> bool:
    return bool(getattr(st, "st_flags", 0) & SF_DATALESS)


def _modell_pruefen(p: Path) -> str | None:
    """None, wenn die Datei ein fertiges whisper.cpp-Modell ist — sonst
    der Grund (für die Einstellungen, nicht für die Auswahl)."""
    try:
        st = p.stat()
        if _dataless(st):
            return "icloud"          # nicht öffnen: das würde blockieren
        if time.time() - st.st_mtime < MODELL_RUHE_S:
            return "kopiert"
        with p.open("rb") as f:
            magic = f.read(4)
    except OSError:
        return "unlesbar"
    if magic != GGML_MAGIC:
        return "kein-ggml"
    if st.st_size < MODELL_MIN_BYTES:
        return "unvollstaendig"
    return None


def get_available_models() -> list[dict]:
    """Mitgelieferte und eigene Modelle als EINE Liste. Bei gleichem Namen
    gewinnt das eigene (so ersetzt man das mitgelieferte `medium` durch
    eine quantisierte Variante). Eigene Modelle werden geprüft (Magic,
    fertig kopiert); was durchfällt, steht in `ungueltig`."""
    nach_name: dict[str, dict] = {}
    d = get_models_dir()
    if d.is_dir():
        for p in sorted(d.glob("ggml-*.bin")):
            nach_name[p.stem[5:]] = {
                "name": p.stem[5:], "size_mb": round(p.stat().st_size / 1e6, 1),
                "quelle": "bundled"}
    e = get_eigene_modelle_dir()
    if e is not None and e.is_dir():
        for p in sorted(e.glob("ggml-*.bin")):
            if _modell_pruefen(p) is None:
                nach_name[p.stem[5:]] = {
                    "name": p.stem[5:], "size_mb": round(p.stat().st_size / 1e6, 1),
                    "quelle": "eigen"}
    aus = sorted(nach_name.values(), key=lambda m: m["size_mb"])
    return aus


def get_ungueltige_modelle() -> list[dict]:
    """Dateien im eigenen Ordner, die (noch) kein Modell sind — mit Grund."""
    e = get_eigene_modelle_dir()
    if e is None or not e.is_dir():
        return []
    aus = []
    for p in sorted(e.iterdir()):
        if p.name.startswith(".") or p.is_dir():
            continue
        grund = _modell_pruefen(p) if p.name.startswith("ggml-") and p.suffix == ".bin" \
            else "name"
        if grund is not None:
            aus.append({"datei": p.name, "grund": grund})
    return aus


def model_pfad(model: str) -> Path:
    """Die Datei zum Modellnamen — eigener Ordner zuerst, dann Bundle.
    Dieselbe Regel wie die Auswahl (Review 2026-09-11): eine eigene Datei
    zählt nur, wenn sie die Prüfung besteht — sonst schattete eine halb
    kopierte oder falsche `ggml-medium.bin` das mitgelieferte Modell,
    während die Einstellungen es als gültig zeigen."""
    e = get_eigene_modelle_dir()
    if e is not None:
        p = e / f"ggml-{model}.bin"
        if p.is_file() and _modell_pruefen(p) is None:
            return p
    p = get_models_dir() / f"ggml-{model}.bin"
    if p.is_file():
        return p
    raise FileNotFoundError(f"Modell nicht gefunden: {p}")


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
    # Wieviele Läufe gleichzeitig rechnen dürfen. 1 = nacheinander
    # (User 2026-09-09: „der Batch startet alle zugleich"). Mehr als
    # einer teilt sich dieselbe GPU und dieselben Kerne — vier Läufe
    # parallel sind nicht schneller als vier nacheinander, nur
    # unübersichtlicher.
    "max_parallel": 1,
    # Identität im Dossier (User 2026-09-10, FORMAT.md §3.1): Wer im
    # Journal eines enrich-Dossiers steht. `user_email` ist freiwillig
    # und wird beim Export ins Dossier geschrieben — die Einstellungen
    # sagen das dazu. `install_id` ist eine zufällige Kennung dieser
    # Installation, beim ersten Start erzeugt, in den Einstellungen
    # sichtbar und neu würfelbar — NIE Hardware-UUID oder Hostname.
    "user_email": "",
    "install_id": "",
    # Zotero (2026-09-11): Metadaten aus der LOKALEN zotero.sqlite, nur
    # lesend (immutable), nur auf Anfrage, nur mit Einwilligung — dieselbe
    # Semantik wie enrichs `zotero_consent`. `zotero_dir` leer = suchen
    # (~/Zotero, Zotero-Profil, /Volumes/*/Zotero).
    "zotero_consent": False,
    "zotero_dir": "",
}


def ulid() -> str:
    """26 Zeichen Crockford-Base32: 48 Bit Zeit + 80 Bit Zufall —
    sortierbar, kollisionsfrei, ohne Bezug zu Gerät oder Person
    (FORMAT.md §2: Record-IDs sind `<typ>-<ULID>`)."""
    import secrets
    import time
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    wert = (int(time.time() * 1000) << 80) | secrets.randbits(80)
    zeichen = []
    for _ in range(26):
        zeichen.append(alphabet[wert & 31])
        wert >>= 5
    return "".join(reversed(zeichen))


def neue_install_id() -> str:
    return "ins-" + ulid()


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
    # Installations-Kennung beim ersten Lesen erzeugen und festschreiben
    if not cfg.get("install_id"):
        cfg["install_id"] = neue_install_id()
        try:
            write_config({"install_id": cfg["install_id"]})
        except OSError:
            pass
    return cfg


def identitaet() -> dict:
    """Wer im Journal eines Dossiers steht: App, Installation, Person."""
    cfg = read_config()
    return {"app": f"localtranscript/{APP_VERSION}",
            "install": cfg["install_id"],
            "user": cfg.get("user_email") or None}


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


def max_parallel() -> int:
    """Gleichzeitige Läufe, 1–4. Wird bei JEDEM Slot-Versuch gelesen —
    eine Änderung in den Einstellungen wirkt sofort, auch auf schon
    wartende Jobs."""
    try:
        n = int(read_config().get("max_parallel", 1))
    except (TypeError, ValueError):
        return 1
    return max(1, min(4, n))
