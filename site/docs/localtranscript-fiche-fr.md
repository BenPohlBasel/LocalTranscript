# LocalTranscript — Fiche pour les chercheur·euse·s

Ce qu'est l'application, quelle IA y fait quoi, où se trouve le code,
et pourquoi la transcription ne quitte jamais l'ordinateur. À remettre à
une direction de projet, un comité d'éthique ou des collègues. État au
10 septembre 2026, version 2.2.0.

## Ce que fait l'application

LocalTranscript transforme des enregistrements audio — entretiens,
discussions de groupe, ateliers — en texte avec codes temporels et
attribution des locuteurs. La transcription est ensuite corrigée dans un
éditeur, les locuteurs sont nommés, les noms remplacés, et le résultat
est exporté pour l'analyse (ATLAS.ti, MAXQDA, NVivo via REFI-QDA ;
enrich ; WebVTT, CSV, texte). Tout cela se passe sur votre propre Mac.

## Quelle IA fait quoi

| Composant | Tâche | Origine, licence | S'exécute où |
|---|---|---|---|
| whisper.cpp avec le modèle `large-v3-turbo` | Reconnaissance vocale : audio → texte avec codes temporels | modèle d'OpenAI (MIT), exécution whisper.cpp (MIT) | localement, sur la carte graphique du Mac |
| silero-vad | Activité vocale : détecte où l'on parle | MIT | localement |
| SpeechBrain ECAPA-TDNN | Séparation des locuteurs : calcule des profils de voix et les regroupe par locuteur | Apache-2.0 | localement |

Les trois modèles sont contenus dans le paquet de l'application. Il n'y
a ni accès à un service d'IA, ni compte, ni clé.

**Pas d'IA générative.** Rien n'est résumé, reformulé ni interprété.
L'application livre ce qui a été dit — pas ce qui était voulu.

**Limites de la reconnaissance vocale.** Whisper est un modèle
neuronal. Là où il ne comprend rien (bruits de fond, dialecte,
chevauchements), il peut placer des mots qui n'ont pas été prononcés.
Une transcription issue de LocalTranscript est une **transcription
brute** qui doit être vérifiée contre l'enregistrement ; l'éditeur est
conçu pour cela. L'allemand standard, le français, l'italien et
l'anglais sont bien reconnus, le suisse allemand de façon lacunaire —
les locuteurs sont néanmoins séparés proprement.

## Où se trouve le code

- Code source : <https://github.com/BenPohlBasel/LocalTranscript>
- Licence : AGPL-3.0-or-later — logiciel libre, qui peut être utilisé,
  examiné, modifié et redistribué
- Développé au B/IAS – Basel Institut für angewandte Stadtforschung,
  Beckenweg 6, 4056 Bâle, <https://bias.city>
- Paquet d'installation : signé et notarisé avec un Apple Developer ID,
  somme de contrôle à chaque version sur GitHub
- Qui ne fait pas confiance au binaire : le dépôt contient toute la
  chaîne de construction, l'application peut être compilée soi-même

## La transcription ne quitte pas l'ordinateur

- L'application n'ouvre **aucune connexion réseau sortante** : pas de
  télémétrie, pas de statistiques d'utilisation, pas de vérification de
  mise à jour, pas de rapports de plantage propres.
- Son service interne n'écoute que sur l'adresse de bouclage
  `127.0.0.1` de la machine elle-même et rejette toute requête venue
  d'ailleurs. Le code correspondant se trouve dans
  `backend/src/localtranscript/main.py` — lisible par tout le monde.
- Audio et transcription se trouvent exclusivement dans le dossier de
  bibliothèque choisi. Ce qui est supprimé va dans un dossier corbeille
  à l'intérieur de la bibliothèque, jusqu'à ce qu'il soit vidé.
- Pas de serveur, pas de service cloud, pas de compte utilisateur, pas
  de sous-traitant, pas de transfert vers un pays tiers — parce que rien
  n'est transmis.

Ce que cela ne couvre **pas** : les sauvegardes du Mac (Time Machine,
iCloud Drive pour le dossier Documents) et les données de diagnostic de
macOS suivent les réglages système, pas l'application. Qui place le
dossier de bibliothèque dans un dossier synchronisé synchronise les
enregistrements.

## Anonymiser les noms

- **Renommer les locuteurs :** un nom dans le panneau des locuteurs vaut
  pour tous les segments de cette personne — « Locuteur 1 » devient
  « B3 » en une étape.
- **Noms dans le texte :** « Rechercher et remplacer » trouve un nom
  dans tous les segments, montre chaque occurrence dans son contexte et
  remplace une par une ou toutes d'un coup — même là où la transcription
  a coupé le nom en fin de ligne.
- **Ce que l'application ne décide pas :** quelles informations
  remplacer — lieux, employeurs, événements. Cela reste la décision de
  la personne qui fait la recherche.
- **L'enregistrement reste ce qu'il est.** La pseudonymisation porte
  sur le texte. Les exports REFI-QDA (`.qdpx`) et dossier enrich
  (`.enrich.zip`) contiennent le fichier audio avec la voix et les noms
  réels ; WebVTT, CSV et texte ne contiennent que le texte. Qui ne veut
  transmettre que des données pseudonymisées transmet un format texte.

## Pour la partie méthodologique

> Les enregistrements ont été transcrits avec LocalTranscript 2.2.0
> (B/IAS Bâle, AGPL-3.0 ; reconnaissance vocale whisper.cpp avec le
> modèle large-v3-turbo, séparation des locuteurs avec SpeechBrain
> ECAPA-TDNN) entièrement en local sur un ordinateur du groupe de
> recherche, sans transmission à des services externes. Les
> transcriptions brutes ont ensuite été corrigées contre
> l'enregistrement et pseudonymisées.

---

Source : <https://github.com/BenPohlBasel/LocalTranscript> (dossier
`site/docs`). La fiche peut être utilisée et adaptée librement.
