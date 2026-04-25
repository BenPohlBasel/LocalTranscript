"""
Speaker diarization without a HuggingFace token.

Pipeline:
  1. silero-vad          -> coarse speech regions
  2. Sliding 1.5s/0.75s windows over each speech region
  3. SpeechBrain ECAPA-TDNN  -> 192-dim speaker embedding per window
  4. Cosine-similarity clustering
       - Fixed N (caller passes equal min/max) -> SpectralClustering
       - Otherwise                              -> Agglomerative (cosine, threshold)
  5. Majority vote per speech region            -> SpeakerSegment
  6. Merge consecutive same-speaker regions

Why silero for VAD: SpeechBrain's CRDNN VAD has a torch-compatibility bug in
1.1.0 (`'VAD' object has no attribute 'device_type'`). silero-vad is tiny (~1MB),
fast, MIT-licensed, and equally robust for our use case.

Public API matches the previous pyannote-based implementation:
  diarize_audio(audio_path, min_speakers, max_speakers, threshold) -> list[SpeakerSegment]
  get_speaker_at_time(segments, time) -> str | None
  SpeakerSegment(start, end, speaker)
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import soundfile as sf

from .config import APP_ROOT


@dataclass
class SpeakerSegment:
    start: float  # seconds
    end: float    # seconds
    speaker: str  # e.g., "SPEAKER_00"


# Lazy-loaded models — first call downloads to APP_ROOT/models/speechbrain/.
_vad_model = None
_embedding_model = None


def _models_cache_dir() -> Path:
    cache = APP_ROOT / "models" / "speechbrain"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _device() -> str:
    # SpeechBrain 1.1.0 has a `device_type` attribute bug that fires when MPS
    # is selected via run_opts. Diarization is tiny relative to whisper.cpp
    # transcription, so CPU here is unproblematic.
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_vad():
    global _vad_model
    if _vad_model is None:
        from silero_vad import load_silero_vad
        print("Loading silero-vad...")
        _vad_model = load_silero_vad()
    return _vad_model


def _get_embedder():
    global _embedding_model
    if _embedding_model is None:
        from speechbrain.inference.speaker import EncoderClassifier
        savedir = _models_cache_dir() / "spkrec-ecapa-voxceleb"
        print("Loading SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb)...")
        _embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(savedir),
            run_opts={"device": _device()},
        )
    return _embedding_model


def _vad_speech_regions(audio_path: str, waveform: torch.Tensor | None = None) -> list[tuple[float, float]]:
    """
    Return list of (start, end) seconds for speech regions.

    Audio loading goes through soundfile (see _load_audio_16k) instead of
    silero_vad.read_audio because the latter calls torchaudio.load — which in
    torchaudio 2.11+ requires the torchcodec extension we don't bundle.
    """
    from silero_vad import get_speech_timestamps
    model = _get_vad()
    if waveform is None:
        waveform, _ = _load_audio_16k(audio_path)
    # silero accepts a 1-D float32 tensor at 16 kHz
    audio = waveform.squeeze(0).float()
    ts = get_speech_timestamps(
        audio, model,
        sampling_rate=16000,
        return_seconds=True,
        min_silence_duration_ms=300,
        min_speech_duration_ms=250,
    )
    return [(float(t["start"]), float(t["end"])) for t in ts]


def _windows_for_region(start: float, end: float,
                        win: float = 1.5, hop: float = 0.75) -> list[tuple[float, float]]:
    """Sliding windows over a single speech region."""
    duration = end - start
    if duration < 0.4:
        return []
    if duration <= win:
        return [(start, end)]
    out = []
    cur = start
    while cur + win <= end + 1e-3:
        out.append((cur, cur + win))
        cur += hop
    # Tail piece if there's leftover audio not covered
    if out and out[-1][1] < end - 0.15:
        out.append((max(start, end - win), end))
    return out


def _load_audio_16k(audio_path: str) -> tuple[torch.Tensor, int]:
    """
    Load audio as a (1, N) float32 tensor at 16 kHz mono.

    Uses soundfile (libsndfile) instead of torchaudio.load — torchaudio 2.11+
    requires torchcodec for I/O which we don't ship. The backend's audio
    pre-conversion to 16 kHz mono WAV (via ffmpeg in process_job) means we
    almost always read a file that is already correctly formatted; the resample
    and downmix branches below are kept as a safety net for direct callers.
    """
    data, sr = sf.read(audio_path, always_2d=True, dtype="float32")
    # data is (frames, channels) — convert to (channels, frames)
    waveform = torch.from_numpy(data.T)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if sr != 16000:
        # Lazy import: only needed if caller skips ffmpeg pre-processing.
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    return waveform, sr


def _embed_windows(waveform: torch.Tensor, sr: int,
                   windows: list[tuple[float, float]]) -> np.ndarray:
    """Run ECAPA on each (start, end) window. Returns (N, 192) array."""
    encoder = _get_embedder()
    out = []
    with torch.no_grad():
        for s, e in windows:
            i0 = max(0, int(s * sr))
            i1 = min(waveform.shape[1], int(e * sr))
            chunk = waveform[:, i0:i1]
            if chunk.shape[1] < int(0.2 * sr):
                # Pad short chunks so ECAPA doesn't choke on tiny inputs.
                pad = int(0.2 * sr) - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, pad))
            emb = encoder.encode_batch(chunk).squeeze().detach().cpu().numpy()
            out.append(emb)
    return np.array(out)


def _cluster(embeddings: np.ndarray,
             num_speakers: int | None,
             threshold: float) -> np.ndarray:
    """
    Cluster window embeddings into speaker labels.

    The UI's `threshold` is on a 0.25-0.7 scale matching the legacy pyannote
    pipeline. ECAPA cosine-distance space is tighter, so we remap:
        UI 0.7 (Locker)       -> AHC 0.885  -> 1-2 speakers
        UI 0.5 (Normal)       -> AHC 0.775  -> 2-3 speakers
        UI 0.35 (Streng)      -> AHC 0.693  -> 4-5 speakers
        UI 0.25 (Sehr streng) -> AHC 0.638  -> 6-9 speakers
    """
    from sklearn.cluster import AgglomerativeClustering, SpectralClustering

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    X = embeddings / (norms + 1e-9)
    affinity = np.clip(X @ X.T, 0.0, 1.0)
    distance = 1.0 - affinity
    np.fill_diagonal(distance, 0.0)

    if num_speakers and num_speakers >= 2 and num_speakers < len(embeddings):
        sc = SpectralClustering(
            n_clusters=num_speakers,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=0,
        )
        return sc.fit_predict(affinity)

    ahc_distance = 0.5 + max(0.0, min(0.85, threshold)) * 0.55
    ahc = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=ahc_distance,
    )
    return ahc.fit_predict(distance)


def diarize_audio(
    audio_path: str,
    min_speakers: int = 0,
    max_speakers: int = 0,
    threshold: float = 0.5,
) -> list[SpeakerSegment]:
    """Diarize an audio file. Returns list of SpeakerSegment."""
    audio_path = str(Path(audio_path).resolve())

    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    speaker_info = "auto"
    if min_speakers > 0 and max_speakers > 0:
        speaker_info = f"{min_speakers}-{max_speakers}"
    print(f"Diarizing: {Path(audio_path).name} (speakers: {speaker_info}, threshold: {threshold})")

    waveform, sr = _load_audio_16k(audio_path)

    speech_regions = _vad_speech_regions(audio_path, waveform=waveform)
    if not speech_regions:
        print("VAD found no speech regions.")
        return []
    total_speech = sum(e - s for s, e in speech_regions)
    print(f"VAD: {len(speech_regions)} regions, {total_speech:.1f}s of speech")

    # Build windows + region index
    all_windows: list[tuple[float, float]] = []
    region_of_window: list[int] = []
    for ridx, (s, e) in enumerate(speech_regions):
        for w in _windows_for_region(s, e):
            all_windows.append(w)
            region_of_window.append(ridx)
    if not all_windows:
        return []

    print(f"Embedding {len(all_windows)} windows...")
    embeddings = _embed_windows(waveform, sr, all_windows)
    if len(embeddings) == 0:
        return []

    fixed_n = None
    if min_speakers > 0 and max_speakers == min_speakers:
        fixed_n = min_speakers
    elif max_speakers > 0 and len(embeddings) <= max_speakers:
        fixed_n = max(1, len(embeddings))

    print(f"Clustering {len(embeddings)} embeddings...")
    labels = _cluster(embeddings, fixed_n, threshold)

    # Re-cluster within min/max bounds if AHC fell outside
    n_found = len(set(labels))
    if max_speakers > 0 and n_found > max_speakers:
        labels = _cluster(embeddings, max_speakers, threshold)
    elif min_speakers > 0 and n_found < min_speakers and len(embeddings) >= min_speakers:
        labels = _cluster(embeddings, min_speakers, threshold)

    # Per-region majority vote
    by_region: dict[int, list[int]] = {}
    for w_idx, lab in enumerate(labels):
        by_region.setdefault(region_of_window[w_idx], []).append(int(lab))

    segments: list[SpeakerSegment] = []
    for ridx, (s, e) in enumerate(speech_regions):
        labs = by_region.get(ridx, [])
        if not labs:
            continue
        unique, counts = np.unique(labs, return_counts=True)
        mode_label = int(unique[counts.argmax()])
        segments.append(SpeakerSegment(start=s, end=e, speaker=f"SPEAKER_{mode_label:02d}"))

    # Drop noise clusters: speakers with < 2 s total speech are reassigned to
    # the temporally nearest "real" speaker. This catches stray AHC singletons.
    speaker_total: dict[str, float] = {}
    for seg in segments:
        speaker_total[seg.speaker] = speaker_total.get(seg.speaker, 0.0) + (seg.end - seg.start)
    real_speakers = {sp for sp, dur in speaker_total.items() if dur >= 2.0}

    if real_speakers and len(real_speakers) < len(speaker_total):
        cleaned: list[SpeakerSegment] = []
        for seg in segments:
            if seg.speaker in real_speakers:
                cleaned.append(seg)
                continue
            mid = 0.5 * (seg.start + seg.end)
            best = min(
                (s for s in segments if s.speaker in real_speakers),
                key=lambda s: abs(0.5 * (s.start + s.end) - mid),
                default=None,
            )
            if best is not None:
                cleaned.append(SpeakerSegment(start=seg.start, end=seg.end, speaker=best.speaker))
        segments = cleaned

    # Merge consecutive same-speaker segments separated by short gaps
    merged: list[SpeakerSegment] = []
    for seg in segments:
        if merged and merged[-1].speaker == seg.speaker and seg.start - merged[-1].end < 0.5:
            merged[-1] = SpeakerSegment(
                start=merged[-1].start,
                end=seg.end,
                speaker=seg.speaker,
            )
        else:
            merged.append(seg)

    # Re-label speakers densely (SPEAKER_00, SPEAKER_01, ...) by speech-time,
    # so the dominant speaker is always SPEAKER_00.
    by_time: dict[str, float] = {}
    for seg in merged:
        by_time[seg.speaker] = by_time.get(seg.speaker, 0.0) + (seg.end - seg.start)
    order = sorted(by_time, key=lambda s: -by_time[s])
    relabel = {old: f"SPEAKER_{idx:02d}" for idx, old in enumerate(order)}
    merged = [SpeakerSegment(start=s.start, end=s.end, speaker=relabel[s.speaker]) for s in merged]

    n_speakers = len(set(s.speaker for s in merged))
    print(f"Diarization: {n_speakers} speakers, {len(merged)} segments")
    return merged


def get_speaker_at_time(segments: list[SpeakerSegment], time: float) -> str | None:
    for seg in segments:
        if seg.start <= time <= seg.end:
            return seg.speaker
    return None
