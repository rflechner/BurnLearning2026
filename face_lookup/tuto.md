# De la détection de visages à la reconnaissance faciale avec Rust et Burn

Ce tutoriel est le parcours de réalisation de **BurnLearning2026**. Il accompagne un futur billet de blog en français et propose des étapes à implémenter, avec un résultat vérifiable à chaque étape.

Deux objectifs successifs :

1. **Détection** : trouver les visages dans une image et dessiner leurs rectangles. Nous entraînerons un petit CNN depuis zéro avec Burn et WIDER FACE.
2. **Reconnaissance faciale** : comparer un visage avec ceux de personnes inscrites volontairement et répondre « personne connue » ou « inconnu ». Cette étape nécessite un second modèle.

La webcam et la démonstration web viennent après les images fixes.

## 0. Point de départ réel

À la rédaction de ce document, le 15 novembre 2026 :

- `face_lookup/Cargo.toml` déclare Burn `0.22.1`, avec les fonctionnalités `wgpu` et `cuda`.
- `face_lookup/src/main.rs` réalise une addition de tenseurs sur WGPU ; l'utilisation de CUDA est commentée.
- Aucun entraînement, chargement WIDER FACE ou modèle de reconnaissance n'a été validé dans le cadre de ce tutoriel.
- Ce document ne fournit pas encore les implémentations des commandes de préparation, d'entraînement ou de détection.

Les commandes Cargo de la première étape utilisent le projet existant. Les autres actions décrivent du travail à réaliser ; les architectures et hyperparamètres proposés sont des points de départ expérimentaux.

## 1. Valider l'environnement avant de construire le modèle

Depuis PowerShell :

```powershell
Set-Location C:\dev\BurnLearning2026\face_lookup
rustc --version
cargo --version
cargo check
cargo run
```

L'exemple actuel doit afficher l'équivalent de `[[3, 4], [5, 6]]`. Si la compilation ou WGPU échoue, résoudre ce problème avant de poursuivre. Sous Windows, vérifier notamment les outils de compilation C++ nécessaires à Rust et le pilote graphique.

Commencer avec **un seul backend opérationnel**, WGPU. Tester CUDA séparément ensuite, sans changer simultanément le modèle et le traitement des données. Les fonctionnalités de compilation et les capacités du matériel ne garantissent pas à elles seules qu'un backend fonctionne.

Conserver `Cargo.lock` dans Git pour reproduire les versions résolues. `version = "0.22.1"` n'est pas un verrouillage exact : Cargo peut sélectionner une version compatible. Consigner la version effectivement utilisée, le GPU, le pilote et la version de Rust dans chaque expérience.

Pour l'entraînement, vérifier dans la documentation correspondant à la version résolue les fonctionnalités nécessaires à l'autodifférentiation et, si utilisé, au système `Learner`. Ne pas recopier sans adaptation un exemple Burn 0.18 : les API ont évolué.

**Étape réussie si :** l'exemple existant compile et s'exécute sur le backend choisi, avec les versions enregistrées.

## 2. Organiser le code autour du parcours des données

Faire évoluer progressivement le dossier existant vers cette structure proposée :

```text
BurnLearning2026/
├── tuto.md
├── data/                    # archives, images et annotations locales, hors Git
├── runs/                    # expériences, journaux et checkpoints, hors Git
└── face_lookup/
    ├── Cargo.toml
    ├── Cargo.lock
    ├── src/
    │   ├── main.rs          # arguments et lancement des commandes
    │   ├── lib.rs           # modules réutilisables et testables
    │   ├── data/            # annotations, images, transformations et lots
    │   ├── model/           # petit CNN et tête de détection
    │   ├── train/           # cibles, pertes, optimisation et sauvegarde
    │   └── infer/           # décodage, filtrage et dessin des rectangles
    └── tests/              # petits exemples synthétiques sans photos privées
```

Ajouter plus tard `recognition/` pour l'alignement, les représentations faciales et la galerie de personnes. Garder le calcul indépendant de l'affichage pour pouvoir réutiliser le même pipeline avec une webcam ou une interface web.

Prévoir quatre actions dans la future interface de commande : inspecter les données, entraîner, évaluer et détecter sur une image. Une action non implémentée doit renvoyer une erreur explicite, jamais simuler une prédiction réussie.

**Étape réussie si :** les responsabilités sont séparées et `cargo check` passe après la réorganisation.

## 3. Récupérer WIDER FACE et conserver sa provenance

La source retenue est [CUHK-CSE/wider_face sur Hugging Face](https://huggingface.co/datasets/CUHK-CSE/wider_face). La fiche annonce une licence **CC BY-NC-ND 4.0** et le dépôt expose des archives. Ce dataset n'est pas automatiquement « prêt pour Burn ».

Avant de télécharger : lire la fiche et les conditions des auteurs. Le fichier `LICENSE.md` du projet contient actuellement l'Unlicense ; cela ne change pas les droits sur les données ou sur de futurs poids de modèles. Ne pas placer les archives ou images dans le dépôt public. Vérifier séparément les conditions de publication de poids et d'illustrations du billet, notamment si le blog est monétisé ou contient des images transformées.

Les archives du dépôt sont des fichiers LFS servis par le stockage **Xet** de Hugging Face. Xet découpe les gros fichiers en blocs : le téléchargement est resumable, dédupliqué entre dépôts, et les blocs sont mis en cache localement (`~/.cache/huggingface/xet` sur macOS/Linux, `%USERPROFILE%\.cache\huggingface\xet` sur Windows ; déplaçable via la variable `HF_XET_CACHE`). Aucun réglage n'est nécessaire : dès que le paquet `hf_xet` est installé, le CLI `hf` l'utilise automatiquement. Sans `hf_xet`, le téléchargement fonctionne quand même en HTTP simple (voir le bloc PowerShell plus bas, `HF_HUB_DISABLE_XET=1` force ce mode) ; Xet apporte surtout la vitesse et la reprise sur interruption.

Fichiers du dépôt utiles à la première étape (révision `db171f1b7fedf4d3453e81297ff02f9915356d19`) :

| Fichier | Taille | Contenu |
|---|---|---|
| `data/WIDER_train.zip` | ~1,5 Go | 12 880 images d'entraînement |
| `data/WIDER_val.zip` | ~360 Mo | 3 226 images de validation |
| `data/wider_face_split.zip` | ~3,6 Mo | annotations `wider_face_train_bbx_gt.txt`, `wider_face_val_bbx_gt.txt` |
| `data/WIDER_test.zip` | ~1,8 Go | images de test sans annotations publiées — inutile pour commencer |

Soit environ 1,8 Go compressés pour les trois archives nécessaires ; prévoir aussi l'espace une fois extraites.

### Installer le CLI `hf` avec Xet

**macOS** (au choix) :

```bash
# avec uv (recommandé, Python embarqué, rien d'autre à installer)
uv tool install "huggingface_hub[hf_xet]"
# ou avec pip
python3 -m pip install -U --user "huggingface_hub[hf_xet]"
```

**Windows** (PowerShell, au choix) :

```powershell
# avec uv (winget install astral-sh.uv) — pas besoin de Python système
uv tool install "huggingface_hub[hf_xet]"
# ou avec pip
py -m pip install -U "huggingface_hub[hf_xet]"
```

Sans installation globale, `uvx` exécute le CLI à la volée sur les deux OS : `uvx --from "huggingface_hub[hf_xet]" hf --version`.

### Télécharger le dataset en local (commande identique sur macOS et Windows)

Épingler la révision résolue et télécharger dans `data/raw/<révision>` depuis la racine du projet :

```bash
hf download CUHK-CSE/wider_face \
  --repo-type dataset \
  --revision db171f1b7fedf4d3453e81297ff02f9915356d19 \
  --include "data/WIDER_train.zip" \
  --include "data/WIDER_val.zip" \
  --include "data/wider_face_split.zip" \
  --local-dir "data/raw/db171f1b7fedf4d3453e81297ff02f9915356d19"
```

- `--local-dir` écrit de vrais fichiers (pas de liens symboliques) ; sans cette option, les fichiers vont dans le cache HF (`~/.cache/huggingface/hub`) et les blocs Xet dans `~/.cache/huggingface/xet`.
- `--include` filtre les fichiers (répéter l'option) ; `--dry-run` liste ce qui serait téléchargé sans rien écrire.
- Équivalent Python : `snapshot_download(repo_id="CUHK-CSE/wider_face", repo_type="dataset", revision=..., allow_patterns=["data/*.zip"], local_dir=...)`.

Pendant la rédaction de ce document, cette commande a été exécutée sur macOS avec le petit `data/wider_face_split.zip` : le zip contient bien `wider_face_split/wider_face_train_bbx_gt.txt` et `wider_face_val_bbx_gt.txt`.

Enregistrer ensuite la provenance et extraire.

```bash
# macOS
REV=db171f1b7fedf4d3453e81297ff02f9915356d19
echo "$REV" > "data/raw/$REV/revision.txt"
shasum -a 256 data/raw/$REV/data/*.zip | tee "data/raw/$REV/sha256.txt"
mkdir -p "data/extracted/$REV"
for z in data/raw/$REV/data/*.zip; do unzip -q "$z" -d "data/extracted/$REV"; done
```

```powershell
# Windows
$rev = 'db171f1b7fedf4d3453e81297ff02f9915356d19'
$rev | Set-Content "data/raw/$rev/revision.txt"
Get-FileHash "data/raw/$rev/data/*.zip" -Algorithm SHA256 |
    Select-Object Path, Hash |
    Export-Csv "data/raw/$rev/sha256.csv" -NoTypeInformation
New-Item -ItemType Directory -Force "data/extracted/$rev" | Out-Null
foreach ($z in Get-ChildItem "data/raw/$rev/data/*.zip") {
    Expand-Archive -LiteralPath $z.FullName -DestinationPath "data/extracted/$rev"
}
```

Solution de repli **sans Python ni Xet** : télécharger les archives directement en HTTP. Le bloc suivant résout une révision précise et la conserve ; il télécharge environ 1,8 Go d'archives.

```powershell
Set-Location C:\dev\BurnLearning2026
$ErrorActionPreference = 'Stop'
$datasetRepo = 'CUHK-CSE/wider_face'
$datasetInfo = Invoke-RestMethod "https://huggingface.co/api/datasets/$datasetRepo"
$datasetRevision = $datasetInfo.sha
if ($datasetRevision -notmatch '^[0-9a-f]{40}$') {
    throw 'Révision Hugging Face absente ou inattendue.'
}
# Un dossier par révision évite de mélanger deux téléchargements.
$rawDir = "data/raw/$datasetRevision"
New-Item -ItemType Directory -Force $rawDir | Out-Null
$datasetRevision | Set-Content "$rawDir/revision.txt"
$archiveNames = @('WIDER_train.zip', 'WIDER_val.zip', 'wider_face_split.zip')
foreach ($archiveName in $archiveNames) {
    $archiveUrl = "https://huggingface.co/datasets/$datasetRepo/resolve/$datasetRevision/data/$archiveName"
    Invoke-WebRequest -Uri $archiveUrl -OutFile "$rawDir/$archiveName"
}
Get-FileHash "$rawDir/*.zip" -Algorithm SHA256 |
    Select-Object Path, Hash |
    Export-Csv "$rawDir/sha256.csv" -NoTypeInformation
$extractDir = "data/extracted/$datasetRevision"
New-Item -ItemType Directory -Force $extractDir | Out-Null
foreach ($archiveName in $archiveNames) {
    Expand-Archive -LiteralPath "$rawDir/$archiveName" -DestinationPath $extractDir
}
```

Les empreintes locales servent à retrouver les fichiers utilisés ; elles ne constituent pas une vérification indépendante de leur authenticité. Pour reproduire une expérience, réutiliser sa révision enregistrée au lieu de résoudre à nouveau la dernière version.

Vérifier ensuite la présence de `WIDER_train/images`, `WIDER_val/images` et `wider_face_split`. Conserver aussi une copie de la fiche de données et des conditions applicables à cette révision. En cas d'échec partiel, vérifier les archives avant l'extraction ; ne pas mélanger les fichiers avec une ancienne extraction.

**Étape réussie si :** les archives sont extraites, les chemins identifiés et leur provenance enregistrée. Le jeu de test officiel n'est pas nécessaire pour cette première étape.

## 4. Lire les annotations avant de créer des tenseurs

Implémenter un lecteur Rust des fichiers `wider_face_train_bbx_gt.txt` et `wider_face_val_bbx_gt.txt`.

Le format décrit dans le [chargeur Hugging Face](https://huggingface.co/datasets/CUHK-CSE/wider_face/blob/main/wider_face.py) contient, pour chaque image : son chemin relatif, un nombre de visages, puis des lignes de dix entiers. Les quatre premiers sont `x, y, largeur, hauteur`, en pixels ; les autres décrivent notamment les difficultés et l'indicateur `invalid`. Le cas zéro visage peut comporter une ligne de zéros à consommer.

Définir des structures simples : une image, ses dimensions et une liste de boîtes. Conserver les attributs originaux pour pouvoir changer plus tard la politique de filtrage. Valider les chemins relatifs, l'existence des images, les nombres de champs et les boîtes dégénérées.

Pour les annotations invalides, décider explicitement si elles deviennent des régions ignorées par la perte. Les supprimer de la liste des positifs tout en traitant leur zone comme un fond certain peut introduire de mauvais exemples.

Avant toute intégration Burn :

1. Tester le parseur sur des annotations synthétiques : zéro, un et plusieurs visages, entrée tronquée et boîte invalide.
2. Charger quelques images réelles et vérifier leur correspondance avec les annotations.
3. Dessiner les rectangles d'origine sur environ vingt images variées.
4. Comparer visuellement les coordonnées avec les visages.
5. Produire un bilan : images lues, visages, annotations ignorées et erreurs.

Les rectangles de cette étape sont la **vérité terrain**, pas les prédictions d'un modèle.

**Étape réussie si :** les rectangles d'annotation sont bien placés et les cas particuliers sont traités sans désynchroniser le fichier.

## 5. Préparer les images et les lots Burn

Commencer avec une entrée RGB de **320 × 320 pixels**, proposition à ajuster selon la mémoire et la qualité sur les petits visages.

- Redimensionner en préservant les proportions, puis compléter les marges : c'est le *letterboxing*.
- Appliquer exactement la même transformation aux boîtes et aux régions ignorées.
- Conserver le facteur d'échelle et les marges pour revenir aux coordonnées d'origine.
- Convertir les pixels en flottants, initialement dans `[0, 1]`.
- Former les tenseurs d'images `[lot, canaux, hauteur, largeur]`, donc `[N, 3, 320, 320]`.
- Traiter le nombre variable de visages par image avec une liste par échantillon, puis construire les cibles denses de la tête de détection.

Commencer par un chargeur local et un assembleur de lots explicites. Un éventuel adaptateur vers l'API dataset de Burn devra être compilé et testé sur de vraies images avant d'être annoncé compatible.

Conserver les séparations officielles entraînement/validation. Les augmentations aléatoires sont réservées à l'entraînement. Écrire un test de transformation aller-retour des coordonnées, avec une tolérance définie pour l'arrondi.

**Étape réussie si :** un lot réel traverse le chargement et les transformations, et les rectangles restent corrects après redimensionnement puis retour à l'image initiale.

## 6. Construire un premier CNN de détection léger

Pour apprendre, commencer avec un détecteur dense à une seule échelle. Proposition :

```text
Image RGB 320 × 320
  → convolution 3 × 3, 16 canaux, pas 2 + ReLU
  → convolution 3 × 3, 32 canaux, pas 2 + ReLU
  → convolution 3 × 3, 64 canaux, pas 2 + ReLU
  → convolution 3 × 3, 64 canaux, pas 1 + ReLU
  → tête 1 × 1 : 5 valeurs par cellule
```

Avec un padding approprié, la grille de sortie est de 40 × 40. Chaque cellule propose un score de présence et quatre paramètres de boîte. Le nombre de paramètres sera calculé à partir du modèle réellement implémenté.

Définir le contrat mathématique avant de coder la perte :

- Une boîte est affectée à la cellule contenant son centre.
- Les deux décalages du centre sont exprimés dans la cellule et contraints dans `[0, 1]` au décodage.
- Largeur et hauteur sont exprimées relativement à la taille de l'entrée, avec une transformation garantissant des valeurs positives.
- La présence est un logit pendant l'entraînement, puis un score après sigmoid.
- En cas de plusieurs centres dans la même cellule, appliquer une règle déterministe et compter ces collisions. Cette première tête ne peut pas représenter tous ces visages simultanément.

Les foules et petits visages seront difficiles à cette résolution. Une tête multi-échelle sera une amélioration ultérieure, si les mesures montrent qu'elle est nécessaire.

Initialiser les poids aléatoirement. Vérifier la forme de sortie, l'absence de valeurs non finies, puis la circulation des gradients. Un passage avant réussi prouve seulement que le calcul fonctionne, pas que le modèle détecte un visage.

**Étape réussie si :** une image synthétique traverse le CNN et une étape d'optimisation modifie effectivement ses paramètres.

## 7. Faire apprendre quelques images avant tout entraînement long

Le premier objectif est de **surapprendre volontairement 8 à 16 images** : le modèle doit réussir à mémoriser ce petit ensemble. Choisir d'abord des images simples, sans collisions de centres.

1. Fixer la graine aléatoire et désactiver les augmentations.
2. Construire les cibles positives, négatives et ignorées.
3. Combiner une perte de présence adaptée au déséquilibre fond/visage et une perte de localisation, par exemple Smooth L1 sur les cellules positives.
4. Normaliser explicitement les pertes ; éviter toute division par zéro pour une image sans positif.
5. Essayer Adam avec un taux initial de `1e-3`, sans considérer cette valeur comme validée.
6. Journaliser séparément les deux pertes et le nombre de positifs.
7. Sauvegarder régulièrement des prédictions sur les mêmes images.

Utiliser le mécanisme d'autodifférentiation de la version Burn retenue. Le [guide de boucle d'entraînement personnalisée](https://burn.dev/books/burn/custom-training-loop.html) explique la structure générale, mais les signatures doivent être vérifiées dans la version du projet.

Si le modèle ne mémorise pas ce petit ensemble, inspecter d'abord les cibles, les coordonnées, le décodage, les gradients et les masques. Ajouter des milliers d'images ne répare pas une erreur de pipeline.

**Étape réussie si :** les pertes diminuent et les boîtes mémorisées se placent correctement. Ce résultat n'est pas une mesure de généralisation.

## 8. Entraîner et évaluer sur des images différentes

Passer d'abord à un sous-ensemble reproductible, puis au jeu d'entraînement complet. Ajouter progressivement les augmentations, en vérifiant leur effet sur les boîtes. La validation reste déterministe et séparée de l'optimisation.

Chaque dossier d'expérience doit conserver : configuration, graine, version du code, révision des données, politique de filtrage, backend, pertes, mesures et poids. Une sauvegarde destinée à reprendre l'entraînement doit aussi préserver l'état de l'optimiseur et, si utilisé, du planificateur de taux d'apprentissage.

Pour commencer, mesurer précision, rappel et AP à IoU 0,5 avec une correspondance un-à-un entre prédictions et annotations. L'IoU mesure le recouvrement de deux rectangles. Documenter le traitement des régions ignorées et des doublons.

Ces mesures simplifiées ne sont pas automatiquement les résultats officiels WIDER FACE « easy / medium / hard ». Utiliser le protocole officiel et ses annotations d'évaluation pour publier des chiffres comparables au benchmark.

Choisir les paramètres sur la validation, puis garder un ensemble final indépendant pour le bilan. Inclure des images sans visage pour observer les faux positifs, ainsi que des tailles, poses et éclairages variés. Ne pas choisir uniquement les exemples les plus réussis pour le billet.

**Étape réussie si :** un checkpoint rechargé reproduit les résultats et détecte des visages sur des images jamais utilisées pour optimiser ses poids.

## 9. Produire une image avec les rectangles prédits

Implémenter dans `infer/` le parcours complet :

```text
Image → prétraitement → modèle chargé → décodage des boîtes
      → seuil de score → suppression des doublons (NMS)
      → retour aux coordonnées d'origine → dessin → image de sortie
```

La NMS conserve les boîtes les mieux notées lorsque plusieurs couvrent fortement le même visage. Régler ses seuils sur la validation. Borner les rectangles aux dimensions de l'image et gérer proprement l'absence de détection.

Tester au minimum une image sans visage, un portrait, plusieurs personnes et une image non carrée. Reprendre exactement la normalisation utilisée pendant l'entraînement.

**Étape réussie si :** on peut fournir une image et un checkpoint, puis obtenir des rectangles cohérents sans accès aux annotations. C'est le premier objectif publiable du projet.

## 10. Comprendre ce qu'il faut ajouter pour reconnaître une personne

Le détecteur répond « où est un visage ? ». Il ne sait pas « de qui s'agit-il ? ». WIDER FACE fournit des boîtes et des attributs, pas les étiquettes d'identité nécessaires à ce second apprentissage.

La reconnaissance suit un autre pipeline :

```text
Visage détecté → points de repère → visage aligné
              → modèle de représentation faciale → vecteur normalisé
              → comparaison à une galerie → identité connue ou inconnu
```

Le vecteur, souvent appelé *embedding*, résume des caractéristiques utiles à la comparaison. Ce n'est ni une boîte ni un simple score de présence.

Notre détecteur initial ne fournit pas les points de repère nécessaires à un alignement précis. Il faudra ajouter un modèle de repères ou entraîner une tête supplémentaire avec des annotations appropriées. Un recadrage rectangulaire seul ne remplace pas forcément le prétraitement attendu par un modèle de reconnaissance.

## 11. Première reconnaissance : utiliser un modèle de représentation existant

Pour obtenir une première démonstration réaliste, conserver le détecteur entraîné avec Burn et essayer un modèle de reconnaissance préentraîné, par exemple **SFace**. Le billet devra distinguer clairement les poids appris dans notre projet et ceux réutilisés.

Le [tutoriel officiel OpenCV](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html) fournit une référence pour l'alignement et la comparaison. Il ne prouve pas que le modèle s'importe dans Burn.

Procéder ainsi :

1. Choisir un fichier précis et enregistrer sa provenance, son empreinte et sa licence.
2. Reproduire son prétraitement exact : repères, alignement, dimensions, ordre des canaux et normalisation.
3. Obtenir des sorties de référence sur quelques images autorisées.
4. Essayer l'import ONNX dans une version compatible de `burn-onnx` et Burn.
5. Comparer les vecteurs et scores obtenus avec la référence, selon une tolérance documentée.
6. Si l'import échoue, isoler l'opérateur problématique. Un moteur externe peut servir de référence temporaire, mais la démonstration ne sera alors pas entièrement exécutée par Burn.

Le [guide d'import Burn](https://burn.dev/books/burn/onnx-import.html) décrit le mécanisme et ses limites : le support ONNX général ne garantit pas chaque modèle et chaque configuration d'opérateur.

**Étape réussie si :** le même visage produit des représentations cohérentes et l'implémentation reproduit suffisamment la référence sur plusieurs exemples.

## 12. Constituer une galerie et apprendre à dire « inconnu »

Commencer par soi-même et quelques volontaires. À titre de protocole pilote, collecter 10 à 20 photos par personne lors de plusieurs séances, avec différentes lumières, expressions et accessoires. Ce volume ne garantit aucune performance.

Séparer trois usages :

- **Inscription** : photos qui servent à construire la galerie.
- **Calibration** : autres séances pour choisir les seuils, incluant des personnes absentes de la galerie.
- **Évaluation finale** : nouvelles séances gardées de côté jusqu'au bilan.

Éviter de répartir aléatoirement des images consécutives d'une même vidéo entre ces ensembles : elles sont trop similaires.

Normaliser les vecteurs en norme L2, puis comparer par similarité cosinus. On peut conserver plusieurs vecteurs par personne ou moyenner ses vecteurs et renormaliser le résultat. Choisir cette règle sur la calibration.

Ne pas toujours retourner la personne la plus proche. Si le meilleur score est inférieur au seuil retenu, répondre « inconnu ». En cas de scores trop proches entre deux personnes, prévoir également un rejet pour ambiguïté. Un score cosinus n'est pas une probabilité d'identité.

Mesurer les fausses acceptations, les faux rejets et, en identification à plusieurs personnes, les erreurs d'identité. Le seuil dépend du modèle, des photos et de la taille de la galerie : ne pas copier celui d'un autre benchmark comme une garantie.

Conserver localement les photos et vecteurs, les exclure de Git et permettre leur suppression. Une démonstration reconnaissant un visage n'est pas un mécanisme d'authentification : une photo ou une vidéo présentée à la caméra peut la tromper. Ne pas utiliser ce prototype pour déverrouiller une porte.

**Étape réussie si :** des personnes inscrites sont reconnues sur de nouvelles séances, des personnes non inscrites sont rejetées, et les erreurs sont mesurées sur un ensemble indépendant.

## 13. Si l'on veut aussi entraîner la reconnaissance depuis zéro

Traiter cette ambition comme une deuxième expérience de recherche, beaucoup plus exigeante que la petite démonstration de détection.

Il faudra obtenir un dataset avec identités et plusieurs images par identité, dont les droits autorisent l'usage prévu. Quelques photos personnelles ne suffisent pas à apprendre une représentation générale robuste.

Construire un CNN qui produit un vecteur, puis implémenter une méthode d'apprentissage des identités, par exemple une perte à marge angulaire de type ArcFace. Cela implique également de maîtriser la normalisation, la constitution des lots, les identités sous-représentées et les exemples difficiles.

Pour évaluer la généralisation à de nouvelles personnes, séparer les identités d'entraînement de celles de validation et de test. Pour les identités de test, créer ensuite des photos d'inscription et des photos de comparaison distinctes. Comparer à une référence préentraînée avec le même protocole.

Un classifieur entraîné à répondre uniquement « moi / pas moi » sur quelques photos constitue une expérience plus étroite ; il ne remplace pas un modèle général de reconnaissance faciale.

**Étape réussie si :** les comparaisons fonctionnent sur des identités absentes de l'entraînement, avec des mesures et des limites documentées.

## 14. Ajouter la webcam, puis éventuellement le web

Pour la webcam, réutiliser le pipeline d'image fixe. Mesurer séparément capture, détection, alignement, représentation et affichage. Limiter la file d'attente pour ne pas afficher des images anciennes. Lisser les décisions sur plusieurs images peut réduire le clignotement, sans constituer une protection contre l'usurpation.

Pour le navigateur, vérifier d'abord la compilation et l'exécution d'une inférence minimale avec les opérations requises. Puis tester le chargement des poids, la mémoire, la latence et la cohérence numérique avec la version native. La portabilité WebGPU de Burn ne garantit pas à elle seule le fonctionnement de toute l'application.

Commencer le web par une photo chargée volontairement ; ajouter la caméra seulement ensuite. Ne publier ni galerie personnelle ni photos privées avec la démonstration.

**Étape réussie si :** le résultat reste cohérent avec la version sur image fixe et les performances sont mesurées sur le matériel réellement utilisé.

## 15. Jalons à cocher et matière pour le billet

- [ ] J0 — L'exemple de tenseurs fonctionne sur le backend choisi.
- [ ] J1 — Les annotations WIDER FACE sont lues et dessinées correctement.
- [ ] J2 — Les transformations et un lot Burn réel sont vérifiés.
- [ ] J3 — Le CNN propage les gradients et mémorise quelques images.
- [ ] J4 — Le détecteur est évalué sur des images indépendantes.
- [ ] J5 — Une image de sortie contient les rectangles prédits.
- [ ] J6 — Alignement et modèle de représentation sont validés.
- [ ] J7 — Galerie, rejet des inconnus et seuils sont évalués.
- [ ] J8 — Webcam utilisable ; web exploré séparément.
- [ ] Option — Reconnaissance entraînée depuis zéro avec un protocole dédié.

Pour chaque jalon, conserver une commande reproductible, la configuration, le temps mesuré, un résultat représentatif et un échec instructif. Le billet pourra raconter successivement le chargement des données, les erreurs de coordonnées, le premier surapprentissage, les limites sur les petits visages et la différence entre détection et identité.

**La prochaine action concrète est J0, puis l'affichage des annotations de J1.** Le premier livrable utile sera une image annotée correctement ; l'entraînement viendra après cette vérification.

## Références

- [Burn Book](https://burn.dev/books/burn/) — vérifier les API avec la version réellement résolue dans le projet.
- [Boucle d'entraînement Burn](https://burn.dev/books/burn/custom-training-loop.html).
- [Import ONNX dans Burn](https://burn.dev/books/burn/onnx-import.html).
- [Fiche WIDER FACE sur Hugging Face](https://huggingface.co/datasets/CUHK-CSE/wider_face).
- [Xet sur le Hub Hugging Face](https://huggingface.co/docs/hub/en/xet) — stockage par blocs des gros fichiers LFS, utilisé automatiquement par le CLI `hf` quand `hf_xet` est installé.
- [Archives WIDER FACE](https://huggingface.co/datasets/CUHK-CSE/wider_face/tree/main/data).
- [Format lu par le chargeur Hugging Face](https://huggingface.co/datasets/CUHK-CSE/wider_face/blob/main/wider_face.py).
- [Article WIDER FACE, Yang et al., CVPR 2016](https://arxiv.org/abs/1511.06523).
- [Référence OpenCV pour détection et reconnaissance](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html).
- [SFace dans OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) — vérifier les conditions du fichier de modèle effectivement choisi.
