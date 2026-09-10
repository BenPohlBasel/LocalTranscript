# LocalTranscript — Description du traitement des données

Bloc de texte à insérer dans un registre des activités de traitement,
une analyse d'impact relative à la protection des données, une demande
au comité d'éthique ou un plan de gestion des données. État au
10 septembre 2026, LocalTranscript 2.2.0. Les mentions entre
`[crochets]` sont complétées par le responsable du traitement.

Ce texte décrit ce que le logiciel fait et ne fait pas. La
qualification juridique de son propre traitement — au regard du RGPD ou
de la nLPD suisse — revient au responsable du traitement ; le texte ne
remplace pas un avis juridique.

---

## 1. Logiciel utilisé

LocalTranscript, version `[2.2.0]`. Logiciel libre sous
AGPL-3.0-or-later, développé au B/IAS – Basel Institut für angewandte
Stadtforschung. Code source public :
<https://github.com/BenPohlBasel/LocalTranscript>. Le logiciel
fonctionne comme application locale sous macOS (Apple Silicon) et est
installé et exploité par le responsable du traitement lui-même.

## 2. Finalité du traitement

Conversion d'enregistrements audio `[p. ex. entretiens semi-directifs
dans le projet …]` en texte avec codes temporels et attribution des
locuteurs, en vue d'une analyse qualitative ultérieure `[dans ATLAS.ti /
MAXQDA / NVivo / enrich / …]`.

## 3. Personnes concernées et catégories de données

Les personnes concernées sont les personnes enregistrées `[personnes
interrogées, participant·e·s à des discussions de groupe, …]`. Sont
traités les enregistrements vocaux (voix et contenu de l'échange) ainsi
que les transcriptions qui en sont produites, avec codes temporels et
attribution des locuteurs. Selon le contenu, des catégories
particulières de données personnelles peuvent être concernées
`[oui / non : …]`.

## 4. Flux de données d'une transcription

1. **Entrée.** Le fichier audio est lu depuis le système de fichiers
   local de l'appareil (MP3, WAV, M4A, OGG, FLAC).
2. **Traitement.** La reconnaissance vocale (whisper.cpp, modèle
   large-v3-turbo) et la séparation des locuteurs (silero-vad,
   SpeechBrain ECAPA) s'exécutent dans le processus de l'application, sur
   le processeur ou la carte graphique de l'appareil. Tous les modèles
   sont contenus dans le paquet de l'application ; rien n'est téléchargé
   au premier lancement.
3. **Stockage.** Pour chaque transcription, un dossier est créé à
   l'emplacement choisi `[chemin, p. ex. ~/Documents/LocalTranscript]`
   contenant une copie de l'audio, le fichier canonique de transcription
   (JSON), des instantanés d'historique à chaque enregistrement et les
   exports dérivés. Les fichiers de travail temporaires sont supprimés
   après chaque exécution.
4. **Réseau.** Le logiciel n'ouvre aucune connexion réseau sortante :
   pas de télémétrie, pas de statistiques d'utilisation, pas de
   vérification de mise à jour, pas de rapports de plantage propres. Le
   service interne de l'application se lie exclusivement à l'adresse de
   bouclage `127.0.0.1` et rejette les requêtes de tout autre hôte
   (HTTP 421). Les données de diagnostic du système macOS relèvent de
   ses réglages système, non du logiciel.
5. **Sortie.** Les fichiers d'export (WebVTT, CSV, texte brut, REFI-QDA
   `.qdpx`, dossier enrich `.enrich`) sont écrits là où
   l'utilisateur·rice les enregistre. **Les exports REFI-QDA et enrich
   contiennent l'enregistrement audio.** Les transmettre, c'est
   transmettre l'enregistrement.

## 5. Lieu du traitement

Exclusivement sur l'appareil `[appareil, lieu]`, dans la session de la
personne connectée. Il n'y a ni serveur, ni service cloud, ni compte
utilisateur.

## 6. Destinataires, sous-traitants, transferts vers des pays tiers

Aucun. Comme aucune donnée n'est transmise, il n'y a ni destinataire,
ni sous-traitant, ni transfert vers un pays tiers. Les données ne
quittent l'appareil que si le responsable du traitement transmet
lui-même des fichiers d'export `[à …, par …]`.

## 7. Durée de conservation et suppression

Conservation des enregistrements et transcriptions : `[durée, base]`.
Supprimer dans l'application déplace une entrée vers un dossier
corbeille à l'intérieur de la bibliothèque (`_papierkorb`) ; elle n'est
définitivement effacée qu'en vidant ce dossier `[par qui, quand]`. Les
instantanés d'historique se trouvent dans le dossier de la transcription
concernée et sont supprimés avec elle. Les sauvegardes de l'appareil
`[Time Machine, …]` relèvent de la règle de suppression du responsable.

## 8. Mesures techniques et organisationnelles

À la charge du responsable du traitement, le logiciel n'apportant
lui-même aucun contrôle d'accès :

- Chiffrement du disque, p. ex. FileVault : `[actif depuis …]`
- Protection d'accès à l'appareil (connexion, verrouillage d'écran) :
  `[…]`
- Pseudonymisation avant toute transmission — dans l'éditeur de
  l'application, les locuteurs peuvent être renommés et les noms dans le
  texte remplacés par rechercher-remplacer ; la décision de ce qu'il faut
  remplacer revient à la personne qui édite : `[procédure,
  responsabilité]`
- Règle de sauvegarde : `[…]`
- Règle de transmission des fichiers d'export, en particulier ceux
  contenant l'audio : `[…]`

## 9. Base légale et information des personnes concernées

`[consentement / intérêt légitime / privilège de recherche selon … ;
lettre d'information du …]`. Le logiciel n'y contribue en rien.

## 10. Vérifiabilité

Les affirmations de la section 4 peuvent être vérifiées dans le code
source : la liaison du service interne à `127.0.0.1` et le rejet des
hôtes étrangers se trouvent dans
`backend/src/localtranscript/main.py`. Le dépôt contient toute la chaîne
de construction jusqu'au paquet d'installation signé ; qui ne fait pas
confiance au binaire distribué peut le compiler.

---

Source de ce texte : <https://github.com/BenPohlBasel/LocalTranscript>
(dossier `site/docs`). Il peut être utilisé et adapté librement.
