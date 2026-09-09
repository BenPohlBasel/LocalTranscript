"""Batch-Warteschlange: nacheinander statt alle zugleich.

User-Befund 2026-09-09: vier gedroppte Dateien rechneten gleichzeitig
um dieselbe GPU. Vorgabe: der Reihe nach, mit einer Einstellung für
gleichzeitige Läufe, Standard 1.
"""
from __future__ import annotations

import threading
import time

from localtranscript import config, jobs


def _job(nr: int) -> dict:
    j = {"id": f"j{nr}"}
    jobs._CANCEL[j["id"]] = threading.Event()
    return j


def _aufraeumen(*js: dict) -> None:
    for j in js:
        jobs._gib_slot(j)
        jobs._CANCEL.pop(j["id"], None)


def test_standard_ist_einer(monkeypatch):
    monkeypatch.setattr(config, "read_config", lambda: dict(config.DEFAULTS))
    assert config.max_parallel() == 1


def test_zweiter_wartet_bis_der_erste_fertig_ist(monkeypatch):
    monkeypatch.setattr(config, "max_parallel", lambda: 1)
    a, b = _job(1), _job(2)
    jobs._nimm_slot(a)                       # a hat den Platz
    dran = threading.Event()

    def zweiter():
        jobs._nimm_slot(b)
        dran.set()
    t = threading.Thread(target=zweiter, daemon=True)
    t.start()
    assert not dran.wait(0.6), "b lief los, obwohl a den Platz hält"

    jobs._gib_slot(a)                        # a fertig
    assert dran.wait(3), "b bekam den frei gewordenen Platz nicht"
    _aufraeumen(a, b)


def test_zwei_parallel_wenn_erlaubt(monkeypatch):
    monkeypatch.setattr(config, "max_parallel", lambda: 2)
    a, b, c = _job(1), _job(2), _job(3)
    jobs._nimm_slot(a)
    jobs._nimm_slot(b)                       # zweiter Platz frei
    dran = threading.Event()
    threading.Thread(target=lambda: (jobs._nimm_slot(c), dran.set()),
                     daemon=True).start()
    assert not dran.wait(0.6), "c lief bei max_parallel=2 als dritter los"
    jobs._gib_slot(a)
    assert dran.wait(3)
    _aufraeumen(a, b, c)


def test_reihenfolge_bleibt_erhalten(monkeypatch):
    """Wer zuerst gedroppt wurde, kommt zuerst dran — nicht der, den
    das Betriebssystem zufällig zuerst aufweckt."""
    monkeypatch.setattr(config, "max_parallel", lambda: 1)
    halter = _job(0)
    jobs._nimm_slot(halter)
    reihenfolge: list[str] = []
    sperre = threading.Lock()
    js = [_job(i) for i in (1, 2, 3)]

    def warte(j):
        jobs._nimm_slot(j)
        with sperre:
            reihenfolge.append(j["id"])
        jobs._gib_slot(j)

    for j in js:                              # gestaffelt anmelden
        threading.Thread(target=warte, args=(j,), daemon=True).start()
        time.sleep(0.15)
    jobs._gib_slot(halter)
    for _ in range(60):
        if len(reihenfolge) == 3:
            break
        time.sleep(0.1)
    assert reihenfolge == ["j1", "j2", "j3"], reihenfolge
    _aufraeumen(halter, *js)


def test_abbruch_beim_warten_wirkt(monkeypatch):
    """Ein wartender Job muss sich abbrechen lassen, ohne dass jemals
    ein Platz frei wird — sonst hängt „Abbrechen\" bis zum Sankt-
    Nimmerleins-Tag."""
    monkeypatch.setattr(config, "max_parallel", lambda: 1)
    a, b = _job(1), _job(2)
    jobs._nimm_slot(a)
    raus = threading.Event()

    def zweiter():
        try:
            jobs._nimm_slot(b)
        except jobs.Abbruch:
            raus.set()
    threading.Thread(target=zweiter, daemon=True).start()
    time.sleep(0.3)
    jobs._CANCEL[b["id"]].set()
    assert raus.wait(3), "Abbruch erreichte den wartenden Job nicht"
    _aufraeumen(a, b)
