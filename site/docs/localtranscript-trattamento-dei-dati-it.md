# LocalTranscript — Descrizione del trattamento dei dati

Blocco di testo da inserire in un registro delle attività di
trattamento, una valutazione d'impatto sulla protezione dei dati, una
domanda al comitato etico o un piano di gestione dei dati. Stato al
10 settembre 2026, LocalTranscript 2.2.0. Le voci tra `[parentesi
quadre]` vengono completate dal titolare del trattamento.

Nota preliminare: un programma non può essere « conforme al GDPR » —
può esserlo un trattamento. Questo testo descrive che cosa il software
fa e che cosa non fa. Se il trattamento svolto con esso soddisfi il GDPR
o la nLPD svizzera lo giudica il titolare. Il testo non costituisce
consulenza legale.

---

## 1. Software impiegato

LocalTranscript, versione `[2.2.0]`. Software libero sotto
AGPL-3.0-or-later, sviluppato al B/IAS – Basel Institut für angewandte
Stadtforschung. Codice sorgente pubblico:
<https://github.com/BenPohlBasel/LocalTranscript>. Il software gira
come applicazione locale su macOS (Apple Silicon) ed è installato e
gestito dal titolare stesso.

## 2. Finalità del trattamento

Conversione di registrazioni audio `[p. es. interviste semi-strutturate
nel progetto …]` in testo con codici temporali e attribuzione dei
parlanti, ai fini della successiva analisi qualitativa `[in ATLAS.ti /
MAXQDA / NVivo / enrich / …]`.

## 3. Interessati e categorie di dati

Gli interessati sono le persone registrate `[intervistati, partecipanti
a discussioni di gruppo, …]`. Vengono trattate registrazioni vocali
(voce e contenuto della conversazione) e le trascrizioni da esse
prodotte, con codici temporali e attribuzione dei parlanti. A seconda
del contenuto possono essere coinvolte categorie particolari di dati
personali `[sì / no: …]`.

## 4. Flusso dei dati di una trascrizione

1. **Ingresso.** Il file audio viene letto dal file system locale del
   dispositivo (MP3, WAV, M4A, OGG, FLAC).
2. **Elaborazione.** Il riconoscimento vocale (whisper.cpp, modello
   large-v3-turbo) e la separazione dei parlanti (silero-vad, SpeechBrain
   ECAPA) girano nel processo dell'applicazione, sul processore o sulla
   scheda grafica del dispositivo. Tutti i modelli sono contenuti nel
   pacchetto dell'applicazione; al primo avvio non viene scaricato
   nulla.
3. **Archiviazione.** Per ogni trascrizione viene creata una cartella
   nella posizione scelta `[percorso, p. es.
   ~/Documents/LocalTranscript]` con una copia dell'audio, il file
   canonico della trascrizione (JSON), istantanee della cronologia a ogni
   salvataggio e le esportazioni derivate. I file di lavoro temporanei
   vengono rimossi dopo ogni esecuzione.
4. **Rete.** Il software non apre alcuna connessione di rete in uscita:
   niente telemetria, niente statistiche d'uso, niente controllo
   aggiornamenti, nessun rapporto di arresto proprio. Il servizio interno
   dell'applicazione si collega esclusivamente all'indirizzo di loopback
   `127.0.0.1` e rifiuta le richieste di qualsiasi altro host
   (HTTP 421). I dati diagnostici del sistema macOS sono regolati dalle
   sue impostazioni di sistema, non dal software.
5. **Uscita.** I file di esportazione (WebVTT, CSV, testo, REFI-QDA
   `.qdpx`, dossier enrich `.enrich.zip`) vengono scritti dove chi opera
   li salva. **Le esportazioni REFI-QDA ed enrich contengono la
   registrazione audio.** Consegnarle significa consegnare la
   registrazione.

## 5. Luogo del trattamento

Esclusivamente sul dispositivo `[dispositivo, luogo]`, nella sessione
della persona che ha effettuato l'accesso. Non esistono server, servizi
cloud né account utente.

## 6. Destinatari, responsabili del trattamento, trasferimenti verso paesi terzi

Nessuno. Poiché nessun dato viene trasmesso, non vi sono destinatari,
responsabili del trattamento né trasferimenti verso paesi terzi. I dati
lasciano il dispositivo solo se il titolare stesso consegna file di
esportazione `[a …, tramite …]`.

## 7. Conservazione e cancellazione

Conservazione di registrazioni e trascrizioni: `[durata, base]`.
Eliminare nell'applicazione sposta una voce in una cartella cestino
all'interno della libreria (`_papierkorb`); viene rimossa
definitivamente solo svuotando quella cartella `[da chi, quando]`. Le
istantanee della cronologia si trovano nella cartella della rispettiva
trascrizione e vengono cancellate con essa. Le copie di sicurezza del
dispositivo `[Time Machine, …]` seguono la regola di cancellazione del
titolare.

## 8. Misure tecniche e organizzative

A carico del titolare, poiché il software non porta con sé alcun
controllo degli accessi:

- Cifratura del disco, p. es. FileVault: `[attiva dal …]`
- Protezione dell'accesso al dispositivo (login, blocco schermo): `[…]`
- Pseudonimizzazione prima di ogni consegna — nell'editor
  dell'applicazione i parlanti possono essere rinominati e i nomi nel
  testo sostituiti con cerca-e-sostituisci; la decisione su che cosa
  sostituire spetta a chi corregge: `[procedura, responsabilità]`
- Regola per le copie di sicurezza: `[…]`
- Regola per la consegna dei file di esportazione, in particolare di
  quelli con audio: `[…]`

## 9. Base giuridica e informazione degli interessati

`[consenso / interesse legittimo / privilegio di ricerca ai sensi di …;
lettera informativa del …]`. Il software non contribuisce in alcun modo.

## 10. Verificabilità

Le affermazioni della sezione 4 sono verificabili nel codice sorgente:
il collegamento del servizio interno a `127.0.0.1` e il rifiuto degli
host estranei si trovano in `backend/src/localtranscript/main.py`. Il
repository contiene l'intera catena di build fino al pacchetto
d'installazione firmato; chi non si fida del binario distribuito può
compilarlo.

---

Fonte di questo testo: <https://github.com/BenPohlBasel/LocalTranscript>
(cartella `site/docs`). Può essere usato e adattato liberamente.
