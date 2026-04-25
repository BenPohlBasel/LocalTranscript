"""
Glossary candidate extraction for the term-correction modal.

Given a German transcript, return a frequency-sorted list of "interesting"
words: proper nouns, named entities, and rare capitalised tokens. Users
review and rewrite the spellings (e.g., "Denkstadt" -> "Denkstatt"); the
rewrite is then applied to VTT/TXT/CSV with word-boundary regex.

Design notes:
  - spaCy NER catches multi-token names ("Lena Wohlfahrt", "Stadt Bern")
    and tags them PER/ORG/LOC/MISC.
  - Single-token PROPN tokens that NER missed are still captured.
  - Common nouns mis-tagged as LOC (e.g. "Stadt") are filtered by a small
    German stop list + a "is OOV" heuristic.

The first call loads `de_core_news_sm` (lazy + cached). Model has to be
present in the venv (~12 MB) — installed via the requirements wheel URL.
"""

from collections import Counter
from typing import Iterable

# Common German nouns that spaCy occasionally mis-labels as LOC/PER. Drop
# them from the candidate list since they're never what the user wants
# to rename.
_COMMON_DROP = {
    # Generic locations / spaces
    "stadt", "land", "ort", "platz", "raum", "räume", "büro", "büros",
    "haus", "weg", "areal", "areals", "gebiet", "viertel",
    # Common nouns mis-tagged as proper nouns / locations
    "name", "menschen", "punkt", "ziel", "moment", "thema", "frage",
    "antwort", "akteur", "akteure", "akteuren", "akteurs", "akteurinnen",
    "leute", "person", "personen", "öffentlichkeit",
    # Filler / interjection
    "ja", "nein", "okay", "hey", "spass", "spaß",
    "mein name", "nachname",
    # Adjectives as substantives
    "sozialen", "konkreten", "lokalen", "neuen", "alten", "alle",
    # Time
    "morgen", "abend", "nachmittag", "vormittag",
    "jahr", "jahre", "jahren", "monat", "woche",
    "anfang", "ende", "vergangenheit", "zukunft",
    # Generic process / planning vocabulary
    "prozess", "prozesse", "prozessen", "phase", "phasen",
    "planung", "planungen", "planungsprozess",
    "projekt", "projekte", "projekten",
    "ablauf", "verfahren", "vorgehen", "aufgabe", "aufgaben",
    "verwaltung", "verwaltungen",
    "dialog", "diskussion", "diskussionen", "gespräch", "gespräche",
    "verständnis", "vorstellung", "vorstellungen",
    "modell", "modelle", "modell",
    "konzept", "konzepte", "kontext",
    "maßnahme", "maßnahmen", "massnahme", "massnahmen",
    "bereich", "bereichen", "rahmen", "umfang",
    "bedürfnis", "bedürfnisse", "bedarf", "wunsch", "wünsche",
    "bedeutung", "wirkung", "wirkungen",
    "ergebnis", "ergebnisse", "resultat", "fazit",
    "potenzial", "potenziale", "möglichkeit", "möglichkeiten",
    "praxis", "theorie",
    "kultur", "kulturen", "planungskultur", "umbaukultur",
    "labor", "experiment", "test", "tests",
    "bevölkerung", "öffentlichkeit", "stadtgesellschaft",
    "nutzer", "nutzerin", "nutzerinnen", "nutzung", "nutzungen",
    "partizipation", "beteiligung",
    "situation", "lage", "verhältnis",
    "struktur", "strukturen", "system", "systeme",
    "entwicklung", "entwicklungen", "gestaltung",
    "lernens", "lernen", "lern",
    "podcast", "podcasts",
    "freiraum", "freiräume", "freiraumpotenzial", "freiraumpotenziale",
    "vorstudie", "vorstudien", "studie", "studien",
    "grundsatz", "grundsätzen", "grundsätze",
    "musik",
    # Common single-word verbs/adverbs/proper-mistakes
    "anschluss", "stelle", "weise", "art", "weiter", "bitte", "danke",
    "stunde", "stunden", "sekunde", "minute",
    "viel", "wenig", "mehr",
    "drittel", "hälfte", "viertel",
    "mittel", "geld", "kosten",
}


_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("de_core_news_sm")
    return _nlp


def _keep_entity(text: str, label: str) -> bool:
    t = text.strip().lower()
    if not t or len(t) < 3:
        return False
    if t in _COMMON_DROP:
        return False
    # Filter pure-stopword phrases like "mein name"
    return True


def extract_candidates(text: str, max_results: int = 100) -> list[dict]:
    """
    Return list of `{term, count, kind}` dicts sorted by frequency descending.
    `kind` is one of: PER, ORG, LOC, MISC, PROPN.
    """
    if not text or not text.strip():
        return []

    nlp = _get_nlp()
    doc = nlp(text)

    counters: dict[str, dict] = {}

    # Multi-token NER entities (filter common nouns)
    covered_tokens: set[int] = set()
    for ent in doc.ents:
        if ent.label_ not in ("PER", "ORG", "LOC", "MISC"):
            continue
        ent_text = ent.text.strip()
        if not _keep_entity(ent_text, ent.label_):
            continue
        # Track token indices so we don't double-count single-token PROPNs
        for tok in ent:
            covered_tokens.add(tok.i)
        key = ent_text
        if key not in counters:
            counters[key] = {"term": ent_text, "count": 0, "kind": ent.label_}
        counters[key]["count"] += 1

    # Single-token PROPNs not already covered, plus OOV/rare capitalised words
    for tok in doc:
        if tok.i in covered_tokens or tok.is_stop or tok.is_punct or tok.is_space:
            continue
        if tok.is_digit or len(tok.text) < 3 or not tok.is_alpha:
            continue
        word = tok.text
        if word.lower() in _COMMON_DROP:
            continue
        # Keep if proper noun, OR if it's mid-sentence-capitalised and OOV
        is_propn = tok.pos_ == "PROPN"
        is_unusual_cap = (
            word[0].isupper()
            and not tok.is_sent_start
            and tok.is_oov
            and len(word) >= 5
        )
        if not (is_propn or is_unusual_cap):
            continue
        if word not in counters:
            counters[word] = {"term": word, "count": 0, "kind": "PROPN"}
        counters[word]["count"] += 1

    out = sorted(counters.values(), key=lambda x: (-x["count"], x["term"]))
    return out[:max_results]
