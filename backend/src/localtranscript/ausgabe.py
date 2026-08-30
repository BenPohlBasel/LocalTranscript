"""Export-Bau aus dem KANONISCHEN Segment-Modell (v2: Exporte sind
abgeleitet, nie Quelle — transkript.json ist die Wahrheit).

Formate:
- VTT: WEBVTT, Sprecher-Präfix "Name: " NUR beim Sprecherwechsel
  (LocalTranscript-v1-Konvention; enrich parst genau das), lange Cues
  werden untertitel-gerecht gesplittet (normalize_vtt_cues, aus v1
  merge.py verbatim übernommen).
- CSV: "Time-in","Time-out","Speaker","Text" (QUOTE_ALL) — Timecodes
  IMMER hh:mm:ss (v1 wechselte ab Stunde 1 das Format; User-Regel
  2026-08-30 „unbedingt immer hh:mm:ss").
- TXT: "Name: Absatz"-Blöcke je Sprecher-Turn, ohne Timecodes.

Segment-Form überall: {"start": s, "end": s, "sprecher": name|"" ,
"text": str} — Anzeigename, aufgelöst aus den Sprecher-Entitäten.
"""
from __future__ import annotations

import csv
import io
import re


def format_vtt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def format_hms(seconds: float) -> str:
    """IMMER hh:mm:ss."""
    s = int(seconds)
    h, rest = divmod(s, 3600)
    m, sek = divmod(rest, 60)
    return f"{h:02d}:{m:02d}:{sek:02d}"


def normalize_vtt_cues(cues: list[dict], min_duration: float = 2.0, max_duration: float = 7.0) -> list[dict]:
    """
    Normalize VTT cues for optimal subtitle display.

    1. Sort by start time and fix overlaps (with 10ms gap)
    2. Split cues longer than max_duration at natural break points
    3. Merge very short cues with neighbors (same speaker)

    Each cue has: speaker, start, end, text
    """

    if not cues:
        return []

    # Step 0: Sort by start time and fix overlapping timestamps
    sorted_cues = sorted(cues, key=lambda x: x['start'])

    # Fix overlaps: ensure 10ms gap between cues
    for i in range(1, len(sorted_cues)):
        prev_end = sorted_cues[i-1]['end']
        curr_start = sorted_cues[i]['start']

        if curr_start <= prev_end:
            # Overlap or exact match - create 10ms gap
            sorted_cues[i]['start'] = prev_end + 0.01
            # If this makes the cue invalid (start >= end), adjust end
            if sorted_cues[i]['start'] >= sorted_cues[i]['end']:
                sorted_cues[i]['end'] = sorted_cues[i]['start'] + 0.5  # Minimal 0.5s duration

    cues = sorted_cues

    # Step 1: Split long cues at sentence boundaries
    split_cues = []
    sentence_end = re.compile(r'([.!?])\s+')

    for cue in cues:
        duration = cue['end'] - cue['start']

        if duration <= max_duration:
            split_cues.append(cue)
            continue

        # Try to split at sentence boundaries
        text = cue['text']
        sentences = sentence_end.split(text)

        # Reconstruct sentences (split keeps delimiters)
        parts = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i+1] in '.!?':
                parts.append(sentences[i] + sentences[i+1])
                i += 2
            else:
                if sentences[i].strip():
                    parts.append(sentences[i])
                i += 1

        if len(parts) <= 1:
            # No sentence boundaries - try splitting at commas
            comma_parts = [p.strip() for p in text.split(',') if p.strip()]
            if len(comma_parts) > 1:
                parts = [p + ',' if i < len(comma_parts)-1 else p for i, p in enumerate(comma_parts)]
            else:
                # Last resort: split at word boundaries every ~40 chars
                words = text.split()
                parts = []
                current_part = []
                current_len = 0
                for word in words:
                    current_part.append(word)
                    current_len += len(word) + 1
                    if current_len >= 40:
                        parts.append(' '.join(current_part))
                        current_part = []
                        current_len = 0
                if current_part:
                    parts.append(' '.join(current_part))

        if len(parts) <= 1:
            split_cues.append(cue)
            continue

        # Distribute time proportionally by character count
        total_chars = sum(len(p) for p in parts)
        current_time = cue['start']

        for part in parts:
            part_text = part.strip()
            if not part_text:
                continue

            char_ratio = len(part) / total_chars
            part_duration = duration * char_ratio
            part_end = min(current_time + part_duration, cue['end'])

            split_cues.append({
                'speaker': cue['speaker'],
                'start': current_time,
                'end': part_end,
                'text': part_text
            })
            current_time = part_end

    # Step 2: Merge short cues with same speaker
    merged_cues = []

    for cue in split_cues:
        duration = cue['end'] - cue['start']

        if not merged_cues:
            merged_cues.append(cue.copy())
            continue

        prev = merged_cues[-1]
        prev_duration = prev['end'] - prev['start']
        combined_duration = cue['end'] - prev['start']

        # Merge if: same speaker AND (current is short OR previous is short) AND combined not too long
        same_speaker = cue['speaker'] == prev['speaker']
        current_short = duration < min_duration
        prev_short = prev_duration < min_duration
        combined_ok = combined_duration <= max_duration

        if same_speaker and (current_short or prev_short) and combined_ok:
            # Merge with previous
            prev['end'] = cue['end']
            prev['text'] = prev['text'] + ' ' + cue['text']
        else:
            merged_cues.append(cue.copy())

    return merged_cues

def _turns(segmente: list[dict]) -> list[dict]:
    """Aufeinanderfolgende Segmente desselben Sprechers → EIN Turn."""
    turns: list[dict] = []
    for s in segmente:
        if not s["text"].strip():
            continue
        if turns and turns[-1]["sprecher"] == (s.get("sprecher") or ""):
            turns[-1]["text"] += " " + s["text"].strip()
            turns[-1]["end"] = s["end"]
        else:
            turns.append({"start": s["start"], "end": s["end"],
                          "sprecher": s.get("sprecher") or "",
                          "text": s["text"].strip()})
    return turns


def build_vtt(segmente: list[dict]) -> str:
    """WebVTT-STANDARD (User 2026-08-30): der Sprecher ist ein
    Voice-Tag `<v Name>` je Cue — nie Teil des Textes (das
    "Name: "-Präfix war v1-Konvention). Ohne Schließ-Tag, wie es
    gängige Werkzeuge schreiben; der Import versteht weiterhin
    beide Stile (turns.py-Parser)."""
    cues = [{"speaker": s.get("sprecher") or None, "start": s["start"],
             "end": s["end"], "text": s["text"].strip()}
            for s in segmente if s["text"].strip()]
    cues = normalize_vtt_cues(cues)
    zeilen = ["WEBVTT", ""]
    for i, c in enumerate(cues, 1):
        text = c["text"]
        if c["speaker"]:
            text = f"<v {c['speaker']}>{text}"
        zeit = (f"{format_vtt_timestamp(c['start'])} --> "
                f"{format_vtt_timestamp(c['end'])}")
        zeilen += [str(i), zeit, text, ""]
    return "\n".join(zeilen)


def build_csv(segmente: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_ALL)
    w.writerow(["Time-in", "Time-out", "Speaker", "Text"])
    for t in _turns(segmente):
        w.writerow([format_hms(t["start"]), format_hms(t["end"]),
                    t["sprecher"], " ".join(t["text"].split())])
    return buf.getvalue()


def build_txt(segmente: list[dict]) -> str:
    bloecke = []
    for t in _turns(segmente):
        text = " ".join(t["text"].split())
        bloecke.append(f"{t['sprecher']}: {text}" if t["sprecher"]
                       else text)
    return "\n\n".join(bloecke) + ("\n" if bloecke else "")
