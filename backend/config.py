"""
Configuration for LocalTranscript - handles paths for both development and bundled app
"""

import sys
import shutil
from pathlib import Path


def get_app_root() -> Path:
    """
    Get the application root directory.
    - In bundled app (py2app): Returns Resources directory
    - In development: Returns project root (parent of backend/)
    """
    if getattr(sys, 'frozen', False):
        # Running as bundled app (py2app)
        # For py2app: executable is in Contents/MacOS/, Resources is sibling
        exe_path = Path(sys.executable)
        resources = exe_path.parent.parent / "Resources"
        if resources.exists():
            return resources
        # Fallback for PyInstaller
        if hasattr(sys, '_MEIPASS'):
            return Path(sys._MEIPASS)
        return exe_path.parent
    else:
        # Development mode: backend/config.py -> backend/ -> project root
        return Path(__file__).parent.parent


import os


def is_bundled() -> bool:
    """True if running from a packaged bundle (Electron resources or py2app)."""
    if getattr(sys, 'frozen', False):
        return True
    if os.environ.get('WHISPER_BUNDLED') == '1':
        return True
    # Marker file dropped by electron/scripts/sync-resources.mjs
    return (get_app_root() / 'BUNDLED').exists()


def find_executable(name: str, bundled_path: Path) -> str:
    """
    Find an executable. In bundled mode the shipped binary is preferred so the
    end user doesn't need Homebrew. In dev mode Homebrew/PATH is preferred since
    those tend to track the latest libraries.
    """
    homebrew_paths = [
        f"/opt/homebrew/bin/{name}",  # Apple Silicon
        f"/usr/local/bin/{name}",      # Intel Mac
    ]

    if is_bundled():
        # Bundle first, fall back to system if a binary is missing for some reason.
        if bundled_path.exists():
            return str(bundled_path)
        for path in homebrew_paths:
            if Path(path).exists():
                return path
        system_path = shutil.which(name)
        if system_path:
            return system_path
    else:
        for path in homebrew_paths:
            if Path(path).exists():
                return path
        system_path = shutil.which(name)
        if system_path:
            return system_path
        if bundled_path.exists():
            return str(bundled_path)

    raise FileNotFoundError(
        f"Could not find {name}. "
        f"Install with: brew install {name.replace('-cli', '-cpp') if 'whisper' in name else name}"
    )


# Application root directory
APP_ROOT = get_app_root()

# Directories — bin/ and models/ are read inside APP_ROOT (writable in dev,
# read-only in the bundled .app). Uploads/outputs must always be writable, so
# in bundled mode they live under the user's Application Support folder.
BIN_DIR = APP_ROOT / "bin"
MODELS_DIR = APP_ROOT / "models"

if is_bundled():
    USER_DATA_DIR = Path.home() / "Library" / "Application Support" / "LocalTranscript"
    UPLOADS_DIR = USER_DATA_DIR / "uploads"
    OUTPUTS_DIR = USER_DATA_DIR / "outputs"
else:
    UPLOADS_DIR = APP_ROOT / "uploads"
    OUTPUTS_DIR = APP_ROOT / "outputs"

# Create directories if they don't exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Executables (with fallback to system PATH)
def get_whisper_cli() -> str:
    return find_executable("whisper-cli", BIN_DIR / "whisper-cli")

def get_ffmpeg_cli() -> str:
    return find_executable("ffmpeg", BIN_DIR / "ffmpeg")

# For development: also check ~/whisper-models/
def get_models_dir() -> Path:
    """Get models directory, checking bundled and user home directory"""
    if MODELS_DIR.exists() and any(MODELS_DIR.glob("*.bin")):
        return MODELS_DIR

    # Fallback to user's whisper-models directory
    user_models = Path.home() / "whisper-models"
    if user_models.exists():
        return user_models

    return MODELS_DIR
