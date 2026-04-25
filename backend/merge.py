"""
Merge Whisper transcript with speaker diarization to create VTT with speaker labels.
Now with sentence-level segmentation and word-level speaker precision.
"""

import re
from dataclasses import dataclass
from .transcribe import TranscriptSegment, WordToken
from .diarize import SpeakerSegment


def get_speaker_at_time(diarization: list[SpeakerSegment], time: float) -> str | None:
    """Find which speaker is active at a given timestamp"""
    for seg in diarization:
        if seg.start <= time <= seg.end:
            return seg.speaker
    # If no exact match, find closest
    closest = None
    min_distance = float('inf')
    for seg in diarization:
        mid = (seg.start + seg.end) / 2
        dist = abs(mid - time)
        if dist < min_distance:
            min_distance = dist
            closest = seg.speaker
    return closest


@dataclass
class Sentence:
    """A sentence with timing and speaker info"""
    start: float
    end: float
    text: str
    speaker: str | None


def tokens_to_subtitle_segments(tokens: list[WordToken]) -> list[dict]:
    """
    Group tokens into subtitle-friendly segments.
    CRITICAL: Never break in the middle of a word!

    VTT best practices:
    - Max ~90 characters per segment (soft limit)
    - Max ~15 words per segment (soft limit)
    - Max ~8 seconds duration (soft limit)
    - Break at punctuation (. ! ? , ; :)
    - ONLY break at word boundaries (token starts with space)

    Returns list of {start, end, tokens} dicts.
    """
    if not tokens:
        return []

    # Soft limits - can be exceeded to avoid breaking words
    MAX_CHARS = 90
    MAX_WORDS = 15
    MAX_DURATION = 8.0

    # Hard limits - force break even at word boundary
    HARD_MAX_CHARS = 120
    HARD_MAX_WORDS = 20
    HARD_MAX_DURATION = 12.0

    segments = []
    current_tokens = []
    current_start = None
    current_char_count = 0
    current_word_count = 0

    # Punctuation patterns
    sentence_end = re.compile(r'[.!?]$')
    clause_break = re.compile(r'[,;:]$')

    def is_word_boundary(token):
        """Check if this token starts a new word"""
        return token.is_word_start or token.text.startswith(" ")

    def find_last_word_boundary(token_list):
        """Find index of last token that ends a word (next token starts new word or is last)"""
        if len(token_list) <= 1:
            return len(token_list) - 1
        # Look backwards for a good break point
        for i in range(len(token_list) - 1, 0, -1):
            text = token_list[i].text.strip()
            # Good break: after punctuation
            if sentence_end.search(text) or clause_break.search(text):
                return i
        # Fallback: find last word boundary
        for i in range(len(token_list) - 1, 0, -1):
            # Check if NEXT token would start a new word
            if i + 1 < len(token_list) and is_word_boundary(token_list[i + 1]):
                return i
        return len(token_list) - 1

    for i, token in enumerate(tokens):
        if current_start is None:
            current_start = token.start

        text = token.text
        text_stripped = text.strip()
        is_new_word = is_word_boundary(token)

        # Add token first
        current_tokens.append(token)
        current_char_count += len(text_stripped)
        if is_new_word:
            current_word_count += 1

        # Check for natural break points
        is_sentence_end = bool(sentence_end.search(text_stripped))
        is_clause_end = bool(clause_break.search(text_stripped))

        # Check for pause after this token
        has_pause = False
        if i + 1 < len(tokens):
            gap = tokens[i + 1].start - token.end
            has_pause = gap > 0.5

        # Check limits
        duration = token.end - current_start if current_start else 0
        exceeds_soft = (current_char_count > MAX_CHARS or
                       current_word_count > MAX_WORDS or
                       duration > MAX_DURATION)
        exceeds_hard = (current_char_count > HARD_MAX_CHARS or
                       current_word_count > HARD_MAX_WORDS or
                       duration > HARD_MAX_DURATION)

        # Decision: should we close this segment?
        should_close = False

        # Always close at sentence end
        if is_sentence_end:
            should_close = True
        # Close at clause break if segment is getting long
        elif is_clause_end and exceeds_soft:
            should_close = True
        # Close at pause if segment is getting long
        elif has_pause and exceeds_soft:
            should_close = True
        # Force close at hard limits, but only at word boundary
        elif exceeds_hard:
            # Check if next token starts a new word
            next_is_word_start = (i + 1 < len(tokens) and is_word_boundary(tokens[i + 1]))
            if next_is_word_start or i == len(tokens) - 1:
                should_close = True
        # Close at end of input
        elif i == len(tokens) - 1:
            should_close = True

        if should_close and current_tokens:
            segments.append({
                'start': current_start,
                'end': current_tokens[-1].end,
                'tokens': list(current_tokens)
            })
            current_tokens = []
            current_start = None
            current_char_count = 0
            current_word_count = 0

    return segments


def tokens_to_sentences(tokens: list[WordToken]) -> list[dict]:
    """
    Alias for backward compatibility - now uses subtitle-optimized segmentation.
    """
    return tokens_to_subtitle_segments(tokens)


def get_sentence_speaker(tokens: list[WordToken], diarization: list[SpeakerSegment]) -> str | None:
    """
    Get the dominant speaker for a sentence (who spoke most of it).
    """
    if not diarization:
        return None

    speaker_durations = {}

    for token in tokens:
        mid_time = (token.start + token.end) / 2
        speaker = get_speaker_at_time(diarization, mid_time)
        if speaker:
            duration = token.end - token.start
            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration

    if not speaker_durations:
        return None

    return max(speaker_durations, key=speaker_durations.get)


def tokens_to_text(tokens: list[WordToken]) -> str:
    """
    Convert tokens to readable text with proper spacing.
    """
    if not tokens:
        return ""

    parts = []
    for token in tokens:
        text = token.text
        if text.startswith(" "):
            parts.append(text)
        elif token.is_word_start and parts:
            parts.append(" " + text)
        else:
            parts.append(text)

    return "".join(parts).strip()


def merge_tokens_with_speakers(
    tokens: list[WordToken],
    diarization: list[SpeakerSegment]
) -> str:
    """
    Merge word tokens with speaker diarization.

    NEW STRATEGY: Use diarization segments as the primary structure!
    1. Each diarization segment becomes a VTT cue
    2. Assign words to diarization segments based on timing
    3. Speaker label at start of each speaker's segment

    This guarantees speaker labels are always at the correct positions.
    """
    if not tokens:
        return "WEBVTT\n\n"

    if not diarization:
        # Fallback: no diarization, just build VTT from tokens
        return build_vtt_from_tokens_no_speakers(tokens)

    # Step 1: Assign each token to a diarization segment
    diar_segments = assign_tokens_to_diarization(tokens, diarization)

    if not diar_segments:
        return "WEBVTT\n\n"

    # Step 2: Build VTT - one cue per diarization segment, split if too long
    vtt_lines = ["WEBVTT", ""]
    cue_number = 1
    current_speaker = None

    for diar_seg in diar_segments:
        if not diar_seg['tokens']:
            continue

        text = tokens_to_text(diar_seg['tokens'])
        if not text:
            continue

        speaker = diar_seg['speaker']

        # Add speaker label only at speaker change
        if speaker != current_speaker:
            speaker_name = format_speaker_name(speaker)
            text = f"{speaker_name}: {text}"
            current_speaker = speaker

        # Use actual token timings for more accurate timestamps
        start_time = diar_seg['tokens'][0].start
        end_time = diar_seg['tokens'][-1].end

        vtt_lines.append(str(cue_number))
        vtt_lines.append(f"{format_vtt_timestamp(start_time)} --> {format_vtt_timestamp(end_time)}")
        vtt_lines.append(text)
        vtt_lines.append("")
        cue_number += 1

    return "\n".join(vtt_lines)


def assign_tokens_to_diarization(
    tokens: list[WordToken],
    diarization: list
) -> list[dict]:
    """
    Assign word tokens to diarization segments.

    `diarization` may be a list of SpeakerSegment objects or dicts with
    start/end/speaker keys (e.g. as stored in the job record).

    Each diarization segment gets all tokens that fall within its time range.
    Returns list of {speaker, start, end, tokens} dicts.
    """
    if not tokens or not diarization:
        return []

    def _attr(seg, name):
        return seg[name] if isinstance(seg, dict) else getattr(seg, name)

    # Sort diarization by start time
    sorted_diar = sorted(diarization, key=lambda x: _attr(x, 'start'))

    # For each token, pick the diarization segment whose interval contains its
    # midpoint OR — if it falls into a silence gap between segments — the
    # *nearest* segment by time. Without the fallback, tokens that whisper
    # places inside silero's sub-second silence boundaries get dropped, which
    # cuts off the first words of speaker turns.
    seg_starts = [_attr(s, 'start') for s in sorted_diar]
    seg_ends = [_attr(s, 'end') for s in sorted_diar]

    def best_seg_idx(t_mid: float) -> int:
        best_i = 0
        best_d = float('inf')
        for i, (s, e) in enumerate(zip(seg_starts, seg_ends)):
            if s <= t_mid <= e:
                return i
            d = s - t_mid if t_mid < s else t_mid - e
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    by_seg: dict[int, list] = {}
    for token in tokens:
        t_mid = (token.start + token.end) / 2.0
        by_seg.setdefault(best_seg_idx(t_mid), []).append(token)

    result = []
    for i, diar_seg in enumerate(sorted_diar):
        toks = by_seg.get(i, [])
        if not toks:
            continue
        result.append({
            'speaker': _attr(diar_seg, 'speaker'),
            'start': _attr(diar_seg, 'start'),
            'end': _attr(diar_seg, 'end'),
            'tokens': toks,
        })

    # Merge consecutive segments from same speaker (cleaner output)
    merged = []
    for seg in result:
        if merged and merged[-1]['speaker'] == seg['speaker']:
            merged[-1]['end'] = seg['end']
            merged[-1]['tokens'].extend(seg['tokens'])
        else:
            merged.append(seg)

    return merged


def merge_classic_with_speakers(
    segments: list[TranscriptSegment],
    diarization: list[SpeakerSegment]
) -> str:
    """
    Merge classic Whisper segments with speaker diarization.

    Strategy:
    1. Merge Whisper segments into complete sentences (based on punctuation)
    2. Assign speaker to each sentence
    3. Speaker label only at speaker changes, always at sentence start
    """
    if not segments:
        return "WEBVTT\n\n"

    if not diarization:
        return build_vtt_without_speakers(segments)

    # Step 1: Merge segments into sentences
    sentences = merge_segments_to_sentences(segments)

    def get_dominant_speaker(start: float, end: float) -> str | None:
        """Find which speaker dominates a time range"""
        speaker_times = {}
        for seg in diarization:
            overlap_start = max(start, seg.start)
            overlap_end = min(end, seg.end)
            if overlap_start < overlap_end:
                duration = overlap_end - overlap_start
                speaker_times[seg.speaker] = speaker_times.get(seg.speaker, 0) + duration
        if not speaker_times:
            return None
        return max(speaker_times, key=speaker_times.get)

    # Step 2: Build VTT with speaker labels at sentence boundaries
    vtt_lines = ["WEBVTT", ""]
    current_speaker = None
    cue_number = 1

    for sent in sentences:
        text = sent['text'].strip()
        if not text:
            continue

        speaker = get_dominant_speaker(sent['start'], sent['end'])

        # Add speaker label only at speaker change
        if speaker and speaker != current_speaker:
            speaker_name = format_speaker_name(speaker)
            text = f"{speaker_name}: {text}"
            current_speaker = speaker

        vtt_lines.append(str(cue_number))
        vtt_lines.append(f"{format_vtt_timestamp(sent['start'])} --> {format_vtt_timestamp(sent['end'])}")
        vtt_lines.append(text)
        vtt_lines.append("")
        cue_number += 1

    return "\n".join(vtt_lines)


def merge_segments_to_sentences(segments: list[TranscriptSegment]) -> list[dict]:
    """
    Merge Whisper segments into complete sentences.
    A sentence ends with . ! ? or after a long pause.

    Returns list of {start, end, text} dicts.
    """
    import re

    if not segments:
        return []

    sentence_end_pattern = re.compile(r'[.!?]$')

    sentences = []
    current_text_parts = []
    current_start = None
    current_end = None

    for i, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue

        if current_start is None:
            current_start = seg.start

        current_text_parts.append(text)
        current_end = seg.end

        # Check if this segment ends a sentence
        ends_sentence = bool(sentence_end_pattern.search(text))

        # Check for long pause after this segment
        has_pause = False
        if i + 1 < len(segments):
            gap = segments[i + 1].start - seg.end
            has_pause = gap > 1.0  # 1 second pause

        # Check if sentence is getting too long (max 20 seconds)
        duration = current_end - current_start
        too_long = duration > 20.0

        # Close sentence if it ends or has pause or too long
        if ends_sentence or has_pause or too_long or i == len(segments) - 1:
            full_text = " ".join(current_text_parts)
            # Clean up double spaces
            full_text = " ".join(full_text.split())

            if full_text:
                sentences.append({
                    'start': current_start,
                    'end': current_end,
                    'text': full_text
                })

            current_text_parts = []
            current_start = None
            current_end = None

    return sentences


def merge_classic_to_csv(
    segments: list[TranscriptSegment],
    diarization: list[SpeakerSegment]
) -> str:
    """
    Create CSV with speaker-wise paragraphs from classic segments.
    Each row is a continuous speech segment from one speaker.
    """
    import csv
    from io import StringIO

    if not segments:
        return "Time-in,Time-out,Speaker,Text\n"

    def get_dominant_speaker(start: float, end: float) -> str:
        if not diarization:
            return "Speaker 1"
        speaker_times = {}
        for seg in diarization:
            overlap_start = max(start, seg.start)
            overlap_end = min(end, seg.end)
            if overlap_start < overlap_end:
                duration = overlap_end - overlap_start
                speaker_times[seg.speaker] = speaker_times.get(seg.speaker, 0) + duration
        if not speaker_times:
            return "Unknown"
        return format_speaker_name(max(speaker_times, key=speaker_times.get))

    rows = []
    current_row = None

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue

        speaker = get_dominant_speaker(seg.start, seg.end)

        if current_row is None:
            current_row = {
                "time_in": seg.start,
                "time_out": seg.end,
                "speaker": speaker,
                "text": text
            }
        elif current_row["speaker"] == speaker:
            current_row["time_out"] = seg.end
            current_row["text"] += " " + text
        else:
            rows.append(current_row)
            current_row = {
                "time_in": seg.start,
                "time_out": seg.end,
                "speaker": speaker,
                "text": text
            }

    if current_row:
        rows.append(current_row)

    output = StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerow(["Time-in", "Time-out", "Speaker", "Text"])

    for row in rows:
        writer.writerow([
            format_csv_timestamp(row["time_in"]),
            format_csv_timestamp(row["time_out"]),
            row["speaker"],
            row["text"]
        ])

    return output.getvalue()


def build_vtt_from_tokens_no_speakers(tokens: list[WordToken]) -> str:
    """Build VTT from tokens without speaker labels"""
    sentences = tokens_to_sentences(tokens)

    vtt_lines = ["WEBVTT", ""]
    cue_number = 1

    i = 0
    while i < len(sentences):
        segment_start = sentences[i]['start']
        segment_sentences = [sentences[i]]

        j = i + 1
        while j < len(sentences):
            if sentences[j]['end'] - segment_start > 10.0:
                break
            if len(segment_sentences) >= 3:
                break
            segment_sentences.append(sentences[j])
            j += 1

        segment_end = segment_sentences[-1]['end']
        texts = [tokens_to_text(s['tokens']) for s in segment_sentences]
        segment_text = " ".join(t for t in texts if t)

        if segment_text:
            vtt_lines.append(str(cue_number))
            vtt_lines.append(f"{format_vtt_timestamp(segment_start)} --> {format_vtt_timestamp(segment_end)}")
            vtt_lines.append(segment_text)
            vtt_lines.append("")
            cue_number += 1

        i = j

    return "\n".join(vtt_lines)


def merge_tokens_to_csv(
    tokens: list[WordToken],
    diarization: list[SpeakerSegment]
) -> str:
    """
    Create CSV with speaker-wise paragraphs.
    Each row is a continuous speech segment from one speaker.
    """
    import csv
    from io import StringIO

    if not tokens:
        return "Time-in,Time-out,Speaker,Text\n"

    # Group into sentences first
    sentences = tokens_to_sentences(tokens)

    # Assign speaker to each sentence
    sentence_data = []
    for sent in sentences:
        speaker = get_sentence_speaker(sent['tokens'], diarization)
        text = tokens_to_text(sent['tokens'])
        if text:
            sentence_data.append({
                'start': sent['start'],
                'end': sent['end'],
                'speaker': format_speaker_name(speaker) if speaker else "Speaker 1",
                'text': text
            })

    if not sentence_data:
        return "Time-in,Time-out,Speaker,Text\n"

    # Merge consecutive sentences from same speaker into paragraphs
    rows = []
    current_row = None

    for sent in sentence_data:
        if current_row is None:
            current_row = {
                'time_in': sent['start'],
                'time_out': sent['end'],
                'speaker': sent['speaker'],
                'text': sent['text']
            }
        elif current_row['speaker'] == sent['speaker']:
            # Same speaker - merge
            current_row['time_out'] = sent['end']
            current_row['text'] += " " + sent['text']
        else:
            # New speaker - save and start new
            rows.append(current_row)
            current_row = {
                'time_in': sent['start'],
                'time_out': sent['end'],
                'speaker': sent['speaker'],
                'text': sent['text']
            }

    if current_row:
        rows.append(current_row)

    # Generate CSV
    output = StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerow(["Time-in", "Time-out", "Speaker", "Text"])

    for row in rows:
        writer.writerow([
            format_csv_timestamp(row["time_in"]),
            format_csv_timestamp(row["time_out"]),
            row["speaker"],
            row["text"]
        ])

    return output.getvalue()


# Legacy functions for backward compatibility

def merge_transcript_with_speakers(
    transcript: list[TranscriptSegment],
    diarization: list[SpeakerSegment]
) -> str:
    """Legacy function - use merge_tokens_with_speakers for better results"""
    if not transcript:
        return "WEBVTT\n\n"

    if not diarization:
        return build_vtt_without_speakers(transcript)

    def get_dominant_speaker(start: float, end: float) -> str | None:
        speaker_times = {}
        for seg in diarization:
            overlap_start = max(start, seg.start)
            overlap_end = min(end, seg.end)
            if overlap_start < overlap_end:
                duration = overlap_end - overlap_start
                speaker_times[seg.speaker] = speaker_times.get(seg.speaker, 0) + duration
        if not speaker_times:
            return None
        return max(speaker_times, key=speaker_times.get)

    vtt_lines = ["WEBVTT", ""]
    current_speaker = None
    cue_number = 1

    for seg in transcript:
        text = seg.text.strip()
        if not text:
            continue

        speaker = get_dominant_speaker(seg.start, seg.end)

        if speaker and (speaker != current_speaker or current_speaker is None):
            speaker_name = format_speaker_name(speaker)
            text = f"{speaker_name}: {text}"
            current_speaker = speaker

        vtt_lines.append(str(cue_number))
        vtt_lines.append(f"{format_vtt_timestamp(seg.start)} --> {format_vtt_timestamp(seg.end)}")
        vtt_lines.append(text)
        vtt_lines.append("")
        cue_number += 1

    return "\n".join(vtt_lines)


def build_vtt_without_speakers(transcript: list[TranscriptSegment]) -> str:
    """Build VTT without speaker labels"""
    vtt_lines = ["WEBVTT", ""]
    for i, seg in enumerate(transcript, 1):
        vtt_lines.append(str(i))
        vtt_lines.append(f"{format_vtt_timestamp(seg.start)} --> {format_vtt_timestamp(seg.end)}")
        vtt_lines.append(seg.text.strip())
        vtt_lines.append("")
    return "\n".join(vtt_lines)


def merge_to_csv(
    transcript: list[TranscriptSegment],
    diarization: list[SpeakerSegment]
) -> str:
    """Legacy function for backward compatibility"""
    import csv
    from io import StringIO

    if not transcript:
        return "Time-in,Time-out,Speaker,Text\n"

    def get_dominant_speaker(start: float, end: float) -> str:
        if not diarization:
            return "Speaker 1"
        speaker_times = {}
        for seg in diarization:
            overlap_start = max(start, seg.start)
            overlap_end = min(end, seg.end)
            if overlap_start < overlap_end:
                duration = overlap_end - overlap_start
                speaker_times[seg.speaker] = speaker_times.get(seg.speaker, 0) + duration
        if not speaker_times:
            return "Unknown"
        return format_speaker_name(max(speaker_times, key=speaker_times.get))

    rows = []
    current_row = None

    for seg in transcript:
        text = seg.text.strip()
        if not text:
            continue

        speaker = get_dominant_speaker(seg.start, seg.end)

        if current_row is None:
            current_row = {
                "time_in": seg.start,
                "time_out": seg.end,
                "speaker": speaker,
                "text": text
            }
        elif current_row["speaker"] == speaker:
            current_row["time_out"] = seg.end
            current_row["text"] += " " + text
        else:
            rows.append(current_row)
            current_row = {
                "time_in": seg.start,
                "time_out": seg.end,
                "speaker": speaker,
                "text": text
            }

    if current_row:
        rows.append(current_row)

    output = StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerow(["Time-in", "Time-out", "Speaker", "Text"])

    for row in rows:
        writer.writerow([
            format_csv_timestamp(row["time_in"]),
            format_csv_timestamp(row["time_out"]),
            row["speaker"],
            row["text"]
        ])

    return output.getvalue()


def format_csv_timestamp(seconds: float) -> str:
    """Convert seconds to readable timestamp HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_vtt_timestamp(seconds: float) -> str:
    """Convert seconds to VTT timestamp format HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def format_speaker_name(speaker_id: str) -> str:
    """Convert pyannote speaker ID to readable name"""
    if speaker_id and speaker_id.startswith("SPEAKER_"):
        try:
            num = int(speaker_id.split("_")[1]) + 1
            return f"Speaker {num}"
        except (IndexError, ValueError):
            pass
    return speaker_id or "Unknown"


def build_vtt_from_speaker_segments(speaker_segments: list[dict]) -> str:
    """
    Build VTT from pre-transcribed speaker segments.

    Each segment has: speaker, start, end, text
    Speaker label only at speaker changes.
    """
    if not speaker_segments:
        return "WEBVTT\n\n"

    vtt_lines = ["WEBVTT", ""]
    cue_number = 1
    current_speaker = None

    for seg in speaker_segments:
        text = seg['text'].strip()
        if not text:
            continue

        speaker = seg['speaker']

        # Add speaker label only at speaker change
        if speaker != current_speaker:
            speaker_name = format_speaker_name(speaker)
            text = f"{speaker_name}: {text}"
            current_speaker = speaker

        vtt_lines.append(str(cue_number))
        vtt_lines.append(f"{format_vtt_timestamp(seg['start'])} --> {format_vtt_timestamp(seg['end'])}")
        vtt_lines.append(text)
        vtt_lines.append("")
        cue_number += 1

    return "\n".join(vtt_lines)


def build_csv_from_speaker_segments(speaker_segments: list[dict]) -> str:
    """
    Build CSV from pre-transcribed speaker segments.

    Each segment has: speaker, start, end, text
    """
    import csv
    from io import StringIO

    if not speaker_segments:
        return "Time-in,Time-out,Speaker,Text\n"

    output = StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerow(["Time-in", "Time-out", "Speaker", "Text"])

    for seg in speaker_segments:
        text = seg['text'].strip()
        if not text:
            continue

        writer.writerow([
            format_csv_timestamp(seg['start']),
            format_csv_timestamp(seg['end']),
            format_speaker_name(seg['speaker']),
            text
        ])

    return output.getvalue()


def tokens_to_speaker_segments(tokens: list, diarization: list) -> list[dict]:
    """
    Group word-level tokens (from a single whisper-cli pass) into speaker
    blocks using a diarization timeline.

    Output mirrors the structure expected by
    `build_{vtt,csv,txt}_from_speaker_transcript_segments`:
        [{"speaker": "SPEAKER_00", "segments": [TranscriptSegment, ...]}, ...]

    Each block becomes a single TranscriptSegment with the speaker's full
    text in that block; downstream `normalize_vtt_cues` then splits long
    cues at natural sentence/clause boundaries.
    """
    if not tokens:
        return []

    assigned = assign_tokens_to_diarization(tokens, diarization)
    blocks: list[dict] = []
    for block in assigned:
        toks = block.get('tokens') or []
        if not toks:
            continue
        text = tokens_to_text(toks).strip()
        if not text:
            continue
        block_start = toks[0].start
        block_end = toks[-1].end
        seg = TranscriptSegment(start=block_start, end=block_end, text=text)
        blocks.append({'speaker': block['speaker'], 'segments': [seg]})

    return blocks


def build_vtt_from_speaker_transcript_segments(all_segments: list[dict]) -> str:
    """
    Build VTT from speaker blocks, each containing multiple transcript segments.

    Input: list of {speaker, segments: [TranscriptSegment, ...]}
    Output: VTT with speaker labels at start of each speaker block

    This creates proper VTT cues from Whisper's natural segmentation,
    while maintaining speaker attribution.
    """
    if not all_segments:
        return "WEBVTT\n\n"

    # First, collect all cues with speaker info
    raw_cues = []
    for block in all_segments:
        speaker = block['speaker']
        for seg in block['segments']:
            text = seg.text.strip()
            if text:
                raw_cues.append({
                    'speaker': speaker,
                    'start': seg.start,
                    'end': seg.end,
                    'text': text
                })

    # Normalize: split long cues, merge short cues
    normalized_cues = normalize_vtt_cues(raw_cues)

    # Build VTT with speaker labels at changes
    vtt_lines = ["WEBVTT", ""]
    cue_number = 1
    current_speaker = None

    for cue in normalized_cues:
        text = cue['text']
        speaker = cue['speaker']

        # Add speaker label only at speaker change
        if speaker != current_speaker:
            speaker_name = format_speaker_name(speaker)
            text = f"{speaker_name}: {text}"
            current_speaker = speaker

        vtt_lines.append(str(cue_number))
        vtt_lines.append(f"{format_vtt_timestamp(cue['start'])} --> {format_vtt_timestamp(cue['end'])}")
        vtt_lines.append(text)
        vtt_lines.append("")
        cue_number += 1

    return "\n".join(vtt_lines)


def normalize_vtt_cues(cues: list[dict], min_duration: float = 2.0, max_duration: float = 7.0) -> list[dict]:
    """
    Normalize VTT cues for optimal subtitle display.

    1. Sort by start time and fix overlaps (with 10ms gap)
    2. Split cues longer than max_duration at natural break points
    3. Merge very short cues with neighbors (same speaker)

    Each cue has: speaker, start, end, text
    """
    import re

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


def build_txt_from_speaker_transcript_segments(all_segments: list[dict]) -> str:
    """
    Plain-text export with speaker labels, one paragraph per speaker block.
    No timecodes. Example:
        Speaker 1: Hallo, wie geht es dir?

        Speaker 2: Gut, danke.
    """
    if not all_segments:
        return ""

    paragraphs = []
    for block in all_segments:
        speaker_label = format_speaker_name(block['speaker'])
        texts = [seg.text.strip() for seg in block['segments'] if seg.text.strip()]
        if not texts:
            continue
        full = " ".join(texts)
        full = " ".join(full.split())  # collapse whitespace
        paragraphs.append(f"{speaker_label}: {full}")

    return "\n\n".join(paragraphs) + "\n" if paragraphs else ""


def build_txt_from_transcript_segments_no_speakers(all_segments: list, pause_threshold: float = 1.5) -> str:
    """
    Plain-text export without speaker labels.
    Paragraph breaks inserted on silent gaps longer than pause_threshold seconds.
    """
    if not all_segments:
        return ""

    paragraphs = []
    current = []
    prev_end = None

    for seg in all_segments:
        text = seg.text.strip()
        if not text:
            continue
        if prev_end is not None and (seg.start - prev_end) > pause_threshold and current:
            paragraphs.append(" ".join(current))
            current = []
        current.append(text)
        prev_end = seg.end

    if current:
        paragraphs.append(" ".join(current))

    paragraphs = [" ".join(p.split()) for p in paragraphs if p.strip()]
    return "\n\n".join(paragraphs) + "\n" if paragraphs else ""


def build_vtt_from_transcript_segments_no_speakers(all_segments: list) -> str:
    """
    Build VTT from TranscriptSegment list without speaker labels.
    Reuses normalize_vtt_cues for clean subtitle splitting.
    """
    if not all_segments:
        return "WEBVTT\n\n"

    raw_cues = []
    for seg in all_segments:
        text = seg.text.strip()
        if text:
            raw_cues.append({
                'speaker': None,
                'start': seg.start,
                'end': seg.end,
                'text': text,
            })

    normalized_cues = normalize_vtt_cues(raw_cues)

    vtt_lines = ["WEBVTT", ""]
    cue_number = 1
    for cue in normalized_cues:
        vtt_lines.append(str(cue_number))
        vtt_lines.append(f"{format_vtt_timestamp(cue['start'])} --> {format_vtt_timestamp(cue['end'])}")
        vtt_lines.append(cue['text'])
        vtt_lines.append("")
        cue_number += 1

    return "\n".join(vtt_lines)


def build_csv_from_transcript_segments_no_speakers(all_segments: list) -> str:
    """
    Build CSV from TranscriptSegment list without speaker column.
    Keeps the same column schema as the speaker version so downstream tools
    that expect four columns still work; the Speaker column is left empty.
    """
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerow(["Time-in", "Time-out", "Speaker", "Text"])

    if not all_segments:
        return output.getvalue()

    for seg in all_segments:
        text = seg.text.strip()
        if not text:
            continue
        writer.writerow([
            format_csv_timestamp(seg.start),
            format_csv_timestamp(seg.end),
            "",
            text,
        ])

    return output.getvalue()


def build_csv_from_speaker_transcript_segments(all_segments: list[dict]) -> str:
    """
    Build CSV from speaker blocks, each containing multiple transcript segments.

    Input: list of {speaker, segments: [TranscriptSegment, ...]}
    Output: CSV with one row per speaker block (all text merged)
    """
    import csv
    from io import StringIO

    if not all_segments:
        return "Time-in,Time-out,Speaker,Text\n"

    output = StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerow(["Time-in", "Time-out", "Speaker", "Text"])

    for block in all_segments:
        speaker = block['speaker']
        segments = block['segments']

        if not segments:
            continue

        # Merge all text in this speaker block
        texts = [seg.text.strip() for seg in segments if seg.text.strip()]
        if not texts:
            continue

        full_text = " ".join(texts)
        start_time = segments[0].start
        end_time = segments[-1].end

        writer.writerow([
            format_csv_timestamp(start_time),
            format_csv_timestamp(end_time),
            format_speaker_name(speaker),
            full_text
        ])

    return output.getvalue()
