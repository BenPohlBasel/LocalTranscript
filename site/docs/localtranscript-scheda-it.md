# LocalTranscript — Scheda per ricercatrici e ricercatori

Che cos'è l'app, quale IA al suo interno fa che cosa, dove si trova il
codice e perché la trascrizione non lascia mai il computer. Da
consegnare alla direzione del progetto, al comitato etico o ai colleghi.
Stato al 10 settembre 2026, versione 2.2.0.

## Che cosa fa l'app

LocalTranscript trasforma registrazioni audio — interviste, discussioni
di gruppo, workshop — in testo con codici temporali e attribuzione dei
parlanti. La trascrizione viene poi corretta in un editor, i parlanti
vengono nominati, i nomi sostituiti, e il risultato esportato per
l'analisi (ATLAS.ti, MAXQDA, NVivo tramite REFI-QDA; enrich; WebVTT,
CSV, testo). Tutto avviene sul proprio Mac.

## Quale IA fa che cosa

| Componente | Compito | Origine, licenza | Gira dove |
|---|---|---|---|
| whisper.cpp con il modello `large-v3-turbo` | Riconoscimento vocale: audio → testo con codici temporali | modello di OpenAI (MIT), runtime whisper.cpp (MIT) | in locale, sulla scheda grafica del Mac |
| silero-vad | Attività vocale: rileva dove si parla | MIT | in locale |
| SpeechBrain ECAPA-TDNN | Separazione dei parlanti: calcola profili vocali e li raggruppa per parlante | Apache-2.0 | in locale |

Tutti e tre i modelli sono contenuti nel pacchetto dell'applicazione.
Non c'è accesso a un servizio di IA, nessun account, nessuna chiave.

**Nessuna IA generativa.** Nulla viene riassunto, riformulato o
interpretato. L'app restituisce ciò che è stato detto — non ciò che si
intendeva.

**Limiti del riconoscimento vocale.** Whisper è un modello neurale. Dove
non capisce nulla (rumori di fondo, dialetto, sovrapposizioni) può
inserire parole mai pronunciate. Una trascrizione di LocalTranscript è
una **trascrizione grezza** che va verificata contro la registrazione;
l'editor è fatto per questo. Tedesco standard, francese, italiano e
inglese vengono riconosciuti bene, lo svizzero tedesco in modo lacunoso
— i parlanti vengono comunque separati correttamente.

## Dove si trova il codice

- Codice sorgente: <https://github.com/BenPohlBasel/LocalTranscript>
- Licenza: AGPL-3.0-or-later — software libero, che può essere usato,
  esaminato, modificato e ridistribuito
- Sviluppato al B/IAS – Basel Institut für angewandte Stadtforschung,
  Beckenweg 6, 4056 Basilea, <https://bias.city>
- Pacchetto d'installazione: firmato e notarizzato con Apple Developer
  ID, checksum a ogni release su GitHub
- Chi non si fida del binario: il repository contiene l'intera catena di
  build, l'app può essere compilata da sé

## La trascrizione non lascia il computer

- L'app non apre **alcuna connessione di rete in uscita**: niente
  telemetria, niente statistiche d'uso, niente controllo aggiornamenti,
  nessun rapporto di arresto proprio.
- Il suo servizio interno ascolta solo sull'indirizzo di loopback
  `127.0.0.1` della macchina stessa e rifiuta ogni richiesta
  proveniente da altrove. Il codice si trova in
  `backend/src/localtranscript/main.py` — leggibile da chiunque.
- Audio e trascrizione si trovano esclusivamente nella cartella della
  libreria scelta. Ciò che viene eliminato finisce in una cartella
  cestino all'interno della libreria, finché non viene svuotata.
- Nessun server, nessun servizio cloud, nessun account utente, nessun
  responsabile del trattamento, nessun trasferimento verso paesi terzi
  — perché nulla viene trasmesso.

Che cosa **non** è coperto: le copie di sicurezza del Mac (Time Machine,
iCloud Drive per la cartella Documenti) e i dati diagnostici di macOS
seguono le impostazioni di sistema, non l'app. Chi colloca la cartella
della libreria in una cartella sincronizzata sincronizza le
registrazioni.

## Anonimizzare i nomi

- **Rinominare i parlanti:** un nome nel pannello dei parlanti vale per
  tutti i segmenti di quella persona — « Parlante 1 » diventa « B3 » in
  un solo passaggio.
- **Nomi nel testo:** « Cerca e sostituisci » trova un nome in tutti i
  segmenti, mostra ogni occorrenza nel contesto e sostituisce una alla
  volta o tutte insieme — anche dove la trascrizione ha spezzato il nome
  a fine riga.
- **Che cosa l'app non decide:** quali informazioni sostituire — luoghi,
  datori di lavoro, eventi. Resta una decisione di chi fa ricerca.
- **La registrazione resta quella che è.** La pseudonimizzazione
  riguarda il testo. Le esportazioni REFI-QDA (`.qdpx`) e dossier enrich
  (`.enrich.zip`) contengono il file audio con voce e nomi reali;
  WebVTT, CSV e testo contengono solo il testo. Chi vuole consegnare
  solo dati pseudonimizzati consegna un formato di testo.

## Per la sezione metodologica

> Le registrazioni sono state trascritte con LocalTranscript 2.2.0
> (B/IAS Basilea, AGPL-3.0; riconoscimento vocale whisper.cpp con il
> modello large-v3-turbo, separazione dei parlanti con SpeechBrain
> ECAPA-TDNN) interamente in locale su un computer del gruppo di ricerca,
> senza trasmissione a servizi esterni. Le trascrizioni grezze sono poi
> state corrette contro la registrazione e pseudonimizzate.

---

Fonte: <https://github.com/BenPohlBasel/LocalTranscript> (cartella
`site/docs`). La scheda può essere usata e adattata liberamente.
