"""
Whisper.cpp integration for audio transcription
With word-level timestamps for precise speaker matching
"""

import subprocess
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional

from .config import get_whisper_cli, get_models_dir


def _get_whisper_cli():
    """Get whisper-cli path (cached)"""
    return get_whisper_cli()


def _get_models_dir():
    """Get models directory (cached)"""
    return get_models_dir()


@dataclass
class TranscriptSegment:
    """Legacy segment format for compatibility"""
    start: float  # seconds
    end: float    # seconds
    text: str


@dataclass
class WordToken:
    """Word/token with precise timestamp"""
    start: float  # seconds
    end: float    # seconds
    text: str
    is_word_start: bool  # True if this starts a new word (has leading space)


def transcribe_audio(
    audio_path: str,
    model: str = "medium",
    language: str = "de",
    progress_callback: Optional[Callable[[int, str, str], None]] = None
) -> list[TranscriptSegment]:
    """
    Transcribe audio using whisper.cpp
    Returns list of transcript segments with timestamps
    progress_callback(percent, message, partial_text) is called with real-time progress
    """
    audio_path = Path(audio_path)
    model_path = _get_models_dir() / f"ggml-{model}.bin"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Get word-level tokens
    tokens = transcribe_with_word_timestamps(
        str(audio_path),
        str(model_path),
        language,
        progress_callback
    )

    # Convert tokens to segments (for backward compatibility)
    segments = tokens_to_segments(tokens)

    return segments


def transcribe_with_word_timestamps(
    audio_path: str,
    model_path: str,
    language: str,
    progress_callback: Optional[Callable[[int, str, str], None]] = None
) -> list[WordToken]:
    """
    Get word-level timestamps using whisper-cli with -ml 1
    """
    audio_path = Path(audio_path)

    # Run whisper-cli with max-len 1 for token-level output
    cmd = [
        _get_whisper_cli(),
        "-m", model_path,
        "-l", language,
        "-f", str(audio_path),
        "-ml", "1",        # Max segment length 1 = token level
        "-oj",             # Output JSON
        "--print-progress",
    ]

    print(f"Running whisper (word-level) on: {audio_path}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        partial_text = []

        for line in process.stdout:
            # Parse progress
            progress_match = re.search(r'progress\s*=\s*(\d+)%', line)
            if progress_match and progress_callback:
                whisper_progress = int(progress_match.group(1))
                # Map whisper 0-100% to overall 35-80% (after diarization)
                overall = 35 + int(whisper_progress * 0.45)
                readable = " ".join(partial_text)
                progress_callback(overall, f"Transkribiere... {whisper_progress}%", readable)

            # Capture text for live display.
            # Whisper-CLI's stdout format is `[ts --> ts]  TOKEN` — exactly two
            # spaces between the closing bracket and the token text. With
            # `-ml 1` each TOKEN is a sub-word; word-starts have an *additional*
            # leading space inside TOKEN. Match the fixed 2-space prefix so we
            # preserve the per-token leading-space marker, then concatenate
            # without an extra separator. This reassembles "Bewohner" cleanly
            # instead of producing "Bew ohner".
            text_match = re.search(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\] {2}(.*)', line)
            if text_match:
                text = text_match.group(1).rstrip('\r\n')
                if text.strip():
                    partial_text.append(text)
                    if progress_callback:
                        readable = "".join(partial_text).strip()
                        progress_callback(-1, "", readable)

        process.wait()

        # Parse JSON output
        json_path = Path(str(audio_path) + ".json")
        if not json_path.exists():
            # Try alternate path
            json_path = audio_path.with_suffix(".wav.json")

        if json_path.exists():
            tokens = parse_whisper_json(json_path)
            # Cleanup JSON file
            json_path.unlink()
            return tokens
        else:
            print(f"Warning: JSON output not found at {json_path}")
            return []

    except Exception as e:
        print(f"Transcription error: {e}")
        raise


def parse_whisper_json(json_path: Path) -> list[WordToken]:
    """Parse whisper JSON output into word tokens"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tokens = []
    for seg in data.get("transcription", []):
        text = seg.get("text", "")
        if not text:
            continue

        # Parse timestamps (format: "00:00:00,000" or "00:00:00.000")
        ts_from = seg["timestamps"]["from"].replace(",", ".")
        ts_to = seg["timestamps"]["to"].replace(",", ".")

        start = parse_timestamp(ts_from)
        end = parse_timestamp(ts_to)

        # Tokens starting with space are word starts
        is_word_start = text.startswith(" ") or len(tokens) == 0

        tokens.append(WordToken(
            start=start,
            end=end,
            text=text,
            is_word_start=is_word_start
        ))

    return tokens


def tokens_to_segments(tokens: list[WordToken], max_segment_duration: float = 10.0) -> list[TranscriptSegment]:
    """
    Convert word tokens back to segments for backward compatibility.
    Groups tokens into natural segments based on pauses.
    """
    if not tokens:
        return []

    segments = []
    current_text = []
    current_start = tokens[0].start
    current_end = tokens[0].end

    for i, token in enumerate(tokens):
        # Check for pause (gap > 0.5s indicates natural break)
        if i > 0:
            gap = token.start - tokens[i-1].end
            if gap > 0.5 or (current_end - current_start) > max_segment_duration:
                # Save current segment
                if current_text:
                    text = "".join(current_text).strip()
                    if text:
                        segments.append(TranscriptSegment(
                            start=current_start,
                            end=current_end,
                            text=text
                        ))
                current_text = []
                current_start = token.start

        current_text.append(token.text)
        current_end = token.end

    # Don't forget last segment
    if current_text:
        text = "".join(current_text).strip()
        if text:
            segments.append(TranscriptSegment(
                start=current_start,
                end=current_end,
                text=text
            ))

    return segments


def get_word_tokens(
    audio_path: str,
    model: str = "medium",
    language: str = "de",
    progress_callback: Optional[Callable[[int, str, str], None]] = None
) -> list[WordToken]:
    """
    Get word-level tokens for precise speaker matching.
    This is the new API for word-level timestamps.
    """
    audio_path = Path(audio_path)
    model_path = _get_models_dir() / f"ggml-{model}.bin"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return transcribe_with_word_timestamps(
        str(audio_path),
        str(model_path),
        language,
        progress_callback
    )


def transcribe_classic(
    audio_path: str,
    model: str = "medium",
    language: str = "de",
    progress_callback: Optional[Callable[[int, str, str], None]] = None
) -> list[TranscriptSegment]:
    """
    Classic Whisper transcription with natural segment boundaries.
    Returns segments as Whisper naturally segments them.
    """
    audio_path = Path(audio_path)
    model_path = _get_models_dir() / f"ggml-{model}.bin"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Run whisper-cli with default settings (natural segmentation)
    cmd = [
        _get_whisper_cli(),
        "-m", str(model_path),
        "-l", language,
        "-f", str(audio_path),
        "-oj",             # Output JSON
        "--print-progress",
    ]

    print(f"Running whisper (classic) on: {audio_path}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        partial_text = []

        for line in process.stdout:
            # Parse progress
            progress_match = re.search(r'progress\s*=\s*(\d+)%', line)
            if progress_match and progress_callback:
                whisper_progress = int(progress_match.group(1))
                overall = 35 + int(whisper_progress * 0.45)
                readable = " ".join(partial_text)
                progress_callback(overall, f"Transkribiere... {whisper_progress}%", readable)

            # Capture text for live display
            text_match = re.search(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]\s*(.+)', line)
            if text_match:
                text = text_match.group(1).strip()
                if text:
                    partial_text.append(text)
                    if progress_callback:
                        readable = " ".join(partial_text)
                        progress_callback(-1, "", readable)

        process.wait()

        # Parse JSON output
        json_path = Path(str(audio_path) + ".json")
        if not json_path.exists():
            json_path = audio_path.with_suffix(".wav.json")

        if json_path.exists():
            segments = parse_whisper_json_classic(json_path)
            json_path.unlink()
            return segments
        else:
            print(f"Warning: JSON output not found at {json_path}")
            return []

    except Exception as e:
        print(f"Transcription error: {e}")
        raise


def parse_whisper_json_classic(json_path: Path) -> list[TranscriptSegment]:
    """Parse whisper JSON output into classic segments"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segments = []
    for seg in data.get("transcription", []):
        text = seg.get("text", "").strip()
        if not text:
            continue

        ts_from = seg["timestamps"]["from"].replace(",", ".")
        ts_to = seg["timestamps"]["to"].replace(",", ".")

        start = parse_timestamp(ts_from)
        end = parse_timestamp(ts_to)

        segments.append(TranscriptSegment(
            start=start,
            end=end,
            text=text
        ))

    return segments


def parse_timestamp(ts: str) -> float:
    """Convert HH:MM:SS.mmm to seconds"""
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(ts)


def transcribe_segment(
    audio_path: str,
    model: str = "medium",
    language: str = "de"
) -> str:
    """
    Transcribe a single audio segment and return plain text.
    Used for transcribing individual speaker segments.
    """
    audio_path = Path(audio_path)
    model_path = _get_models_dir() / f"ggml-{model}.bin"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Run whisper-cli - output to stdout as text
    cmd = [
        _get_whisper_cli(),
        "-m", str(model_path),
        "-l", language,
        "-f", str(audio_path),
        "-otxt",           # Output as text
        "--no-timestamps", # No timestamps needed
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        # Read the text output file
        txt_path = Path(str(audio_path) + ".txt")
        if txt_path.exists():
            text = txt_path.read_text(encoding='utf-8').strip()
            txt_path.unlink()  # Cleanup
            return text

        return ""

    except Exception as e:
        print(f"Segment transcription error: {e}")
        return ""


def transcribe_segment_with_timestamps(
    audio_path: str,
    model: str = "medium",
    language: str = "de",
    time_offset: float = 0.0
) -> list[TranscriptSegment]:
    """
    Transcribe a single audio segment and return segments with timestamps.
    Timestamps are adjusted by time_offset to match original audio position.
    """
    audio_path = Path(audio_path)
    model_path = _get_models_dir() / f"ggml-{model}.bin"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Run whisper-cli with JSON output
    cmd = [
        _get_whisper_cli(),
        "-m", str(model_path),
        "-l", language,
        "-f", str(audio_path),
        "-oj",  # Output JSON
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        # Read the JSON output file
        json_path = Path(str(audio_path) + ".json")
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            json_path.unlink()  # Cleanup

            segments = []
            for seg in data.get("transcription", []):
                text = seg.get("text", "").strip()
                if not text:
                    continue

                ts_from = seg["timestamps"]["from"].replace(",", ".")
                ts_to = seg["timestamps"]["to"].replace(",", ".")

                start = parse_timestamp(ts_from) + time_offset
                end = parse_timestamp(ts_to) + time_offset

                segments.append(TranscriptSegment(
                    start=start,
                    end=end,
                    text=text
                ))

            return segments

        return []

    except Exception as e:
        print(f"Segment transcription error: {e}")
        return []
