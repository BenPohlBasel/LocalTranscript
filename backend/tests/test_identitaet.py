"""Identität im Dossier (User 2026-09-10): freiwillige E-Mail als
User-ID der App, plus zufällige Installations-Kennung — beides landet
im enrich-Export, nie eine Geräte-Kennung."""
from __future__ import annotations

import io
import json
import re
import socket
import zipfile

from localtranscript import config, exporte


def _teil(z: zipfile.ZipFile, endung: str) -> bytes:
    return z.read(next(n for n in z.namelist() if n.endswith(endung)))


def test_install_id_entsteht_einmal_und_bleibt(client):
    a = client.get("/api/settings").json()["install_id"]
    b = client.get("/api/settings").json()["install_id"]
    assert re.fullmatch(r"ins-[0-9A-HJKMNP-TV-Z]{26}", a), a
    assert a == b


def test_install_id_nicht_von_aussen_setzbar_nur_neu(client):
    alt = client.get("/api/settings").json()["install_id"]
    assert client.post("/api/settings",
                       json={"install_id": "ins-FREMD"}).json()["install_id"] == alt
    neu = client.post("/api/settings",
                      json={"install_id": "neu"}).json()["install_id"]
    assert neu != alt and neu.startswith("ins-")


def test_email_freiwillig_und_geprueft(client):
    assert client.get("/api/settings").json()["user_email"] == ""
    assert client.post("/api/settings",
                       json={"user_email": "kein at"}).status_code == 422
    r = client.post("/api/settings", json={"user_email": " nora@uni.ch "})
    assert r.json()["user_email"] == "nora@uni.ch"
    assert client.post("/api/settings",
                       json={"user_email": ""}).json()["user_email"] == ""


def test_export_traegt_app_install_und_email(client, eintrag):
    client.post("/api/settings", json={"user_email": "nora@uni.ch"})
    inhalt, _n, _m = exporte.export_bytes(eintrag, "enrich")
    z = zipfile.ZipFile(io.BytesIO(inhalt))
    tj = json.loads(_teil(z, "transcript.json"))
    wer = json.loads(_teil(z, "manifest.json"))["producer"]
    assert wer["user"] == "nora@uni.ch"
    assert wer["app"].startswith("localtranscript/")
    assert wer["install"].startswith("ins-")
    manifest = _teil(z, "manifest.json").decode()
    assert "nora@uni.ch" in manifest          # producer + Gate-Freigaben
    host = socket.gethostname()               # nie eine Geräte-Kennung
    assert host not in manifest and host not in json.dumps(tj)


def test_ohne_email_steht_die_app_im_dossier(client, eintrag):
    inhalt, _n, _m = exporte.export_bytes(eintrag, "enrich")
    z = zipfile.ZipFile(io.BytesIO(inhalt))
    assert "user" not in json.loads(_teil(z, "manifest.json"))["producer"]
    assert f"localtranscript/{config.APP_VERSION}" in _teil(z, "manifest.json").decode()
