"""Whisper-Schleifen (dreimal derselbe Satz hintereinander) werden zu
einem Segment — Sicherheitsnetz hinter der VAD (Befund 2026-09-12)."""
from localtranscript.jobs import schleifen_zusammenziehen


def _s(t, a, b): return {"start": a, "end": b, "sprecher": None, "text": t}


def test_dreifache_wiederholung_wird_eins():
    seg = [_s("Hallo.", 0, 1), _s("Böden.", 1, 2), _s("Böden.", 2, 3), _s("Böden.", 3, 4), _s("Tschüss.", 4, 5)]
    aus = schleifen_zusammenziehen(seg)
    assert [s["text"] for s in aus] == ["Hallo.", "Böden.", "Tschüss."]
    assert aus[1]["start"] == 1 and aus[1]["end"] == 4


def test_zweimal_bleibt_sprache():
    seg = [_s("Ja.", 0, 1), _s("Ja.", 1, 2), _s("Gut.", 2, 3)]
    assert len(schleifen_zusammenziehen(seg)) == 3


def test_leer_und_leere_texte():
    assert schleifen_zusammenziehen([]) == []
    seg = [_s("", 0, 1), _s("", 1, 2), _s("", 2, 3)]
    assert len(schleifen_zusammenziehen(seg)) == 3       # Leeres zählt nicht als Schleife
