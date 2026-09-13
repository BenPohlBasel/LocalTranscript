"""RTTM-Auswertung der Sprechertrennung (SpeakerKit-CLI).

Die CLI selbst wird hier nicht gerufen — geprüft wird, was wir aus ihrer
Ausgabe machen: Reihenfolge, Überlappungsfreiheit (der Job schneidet je
Block einen Clip; überlappende Blöcke hiessen doppelter Text) und die
Abbildung unserer Trennschärfe auf die VBx-Schwelle.
"""
from __future__ import annotations

from localtranscript.diarize import (
    SpeakerSegment,
    _schwelle,
    get_speaker_at_time,
    rttm_lesen,
)


def zeile(start, dauer, wer):
    return f"SPEAKER feld 1 {start} {dauer} <NA> <NA> {wer} <NA> <NA>"


def test_liest_felder_und_sortiert():
    s = rttm_lesen("\n".join([zeile(9.0, 1.0, "B"), zeile(0.0, 2.5, "A")]))
    assert [(x.start, x.end, x.speaker) for x in s] == [
        (0.0, 2.5, "A"), (9.0, 10.0, "B")]


def test_ueberlappung_wird_gekappt():
    s = rttm_lesen("\n".join([zeile(0.0, 2.0, "A"), zeile(1.5, 2.0, "B")]))
    assert [(x.start, x.end, x.speaker) for x in s] == [
        (0.0, 2.0, "A"), (2.0, 3.5, "B")]


def test_ganz_verdeckter_block_faellt_weg():
    s = rttm_lesen("\n".join([zeile(0.0, 5.0, "A"), zeile(1.0, 1.0, "B")]))
    assert len(s) == 1 and s[0].speaker == "A"


def test_muell_und_nullaenge_stoeren_nicht():
    s = rttm_lesen("\n".join(["", "# Kommentar", "SPEAKER zu kurz",
                              zeile("x", 1.0, "A"), zeile(3.0, 0.0, "A"),
                              zeile(3.0, 1.0, "A")]))
    assert len(s) == 1 and s[0].start == 3.0


def test_leere_ausgabe_ergibt_keine_sprecher():
    assert rttm_lesen("") == []


def test_schwelle_bleibt_im_bereich_der_cli():
    assert _schwelle(0.25) == 0.45          # streng → mehr Sprecher
    assert _schwelle(0.7) == 0.75           # locker → weniger
    assert _schwelle(0.1) == _schwelle(0.25) and _schwelle(9.0) == _schwelle(0.7)
    assert _schwelle(0.3) < _schwelle(0.5) < _schwelle(0.65)


def test_sprecher_zum_zeitpunkt():
    s = [SpeakerSegment(0.0, 2.0, "A"), SpeakerSegment(2.0, 4.0, "B")]
    assert get_speaker_at_time(s, 0.0) == "A"
    assert get_speaker_at_time(s, 2.0) == "B"       # Grenze gehört nach rechts
    assert get_speaker_at_time(s, 9.0) is None


def test_leerzeichen_im_dateinamen_verschiebt_nichts():
    """Die CLI schreibt den Dateinamen ins RTTM — «ton 01.wav» ergab
    sonst den Sprecher «<NA>» und lauter falsche Zeiten."""
    s = rttm_lesen("SPEAKER ton 01 1 3.000 1.500 <NA> <NA> B <NA> <NA>")
    assert [(x.start, x.end, x.speaker) for x in s] == [(3.0, 4.5, "B")]
