# BurnLearning2026

> **EN — Learning project, not production-ready.**
> This repository is a personal experiment to learn the Burn deep learning framework
> in Rust. The code, models, and results are experimental and intended for learning,
> not for production use.
>
> **FR — Projet d'apprentissage, non prêt pour la production.**
> Ce dépôt est une expérimentation personnelle pour apprendre à utiliser Burn,
> le framework de deep learning en Rust. Le code, les modèles et les résultats sont
> expérimentaux et destinés à l'apprentissage, pas à un usage en production.

## API REST du dataset d'entraînement

Depuis la racine du dépôt :

```sh
cargo run --manifest-path face_lookup/Cargo.toml
```

Le serveur écoute sur `http://127.0.0.1:3001`. Les annotations sont chargées une
seule fois au démarrage ; les JPEG sont lus à la demande, sans décodage ni modification.
Une annotation invalide empêche le démarrage avec une erreur de parsing.

### Parcourir les images

```sh
curl 'http://127.0.0.1:3001/api/train?page=1&page_size=20'
```

La réponse JSON contient `page`, `page_size`, `total`, `total_pages` et `items`.
Chaque élément expose `id`, `image_file_path`, `image_url` et `faces` (rectangles
et attributs d'origine). `image_url` est relatif à l'adresse du serveur.

Les pages commencent à **1**, les identifiants d'image à **0** (ordre du fichier
d'annotations). Par défaut : page 1, 20 éléments. `page_size` doit être compris
entre 1 et 100. Une pagination invalide renvoie HTTP 400 ; une page au-delà de la
fin renvoie une liste vide. Les identifiants restent stables tant que le fichier
d'annotations ne change pas.

### Récupérer une image

```sh
curl 'http://127.0.0.1:3001/api/train/0/image' -o image.jpg
```

Renvoie le JPEG original avec `Content-Type: image/jpeg`. Un identifiant inconnu
ou un fichier absent renvoie HTTP 404. Seuls les fichiers référencés par les
annotations et situés dans le dossier des images sont accessibles.

### Configuration

Variables d'environnement facultatives :

- `BIND_ADDR` : adresse d'écoute (défaut `127.0.0.1:3001`).
- `ANNOTATIONS_PATH` : chemin du fichier d'annotations WIDER FACE.
- `IMAGES_ROOT` : dossier racine contenant les images (sous-dossiers inclus).
- `VALID_ANNOTATIONS_PATH` : annotations de validation (défaut : `wider_face_val_bbx_gt.txt` du dataset extrait).
- `VALID_IMAGES_ROOT` : dossier des images de validation (défaut : `WIDER_val/images` du dataset extrait).

Par défaut, les chemins pointent vers le dataset extrait dans `face_lookup/data`,
indépendamment du répertoire de lancement. Pour un autre dataset, définir les
deux variables de chemin. Les chemins personnalisés relatifs sont résolus depuis
le répertoire de lancement.

Le test initial de parsing/décodage et de calcul GPU reste disponible :

```sh
cd face_lookup
cargo run -- --inspect
```

Tests de l'API (pagination, annotations, JPEG, erreurs et confinement des chemins) :

```sh
cargo test --manifest-path face_lookup/Cargo.toml
```

## Entraînement avec le tableau de bord Burn

Depuis la racine du dépôt, dans un terminal interactif :

```sh
cargo run --release --manifest-path face_lookup/Cargo.toml -- --train
```

Un terminal d'environ 120 colonnes sur 40 lignes permet de lire toutes les métriques.

Dans RustRover, la configuration Cargo utilise `run --release -- --train`, avec
le répertoire de travail `face_lookup` et **Emulate terminal in output console**
activé. Les archives `WIDER_train.zip` et `WIDER_val.zip` doivent être extraites dans
`face_lookup/data/extracted/db171f1b7fedf4d3453e81297ff02f9915356d19/`, avec les dossiers
`WIDER_train/images`, `WIDER_val/images` et `wider_face_split`.
Si les photos sont ailleurs, définir `IMAGES_ROOT` et `VALID_IMAGES_ROOT` dans les
variables d'environnement de la configuration. Les deux jeux sont vérifiés avant
d'initialiser le GPU et d'ouvrir l'interface.

Le tableau de bord natif de Burn affiche la progression des époques et les courbes
de **loss totale**, **présence**, **localisation brute**, **localisation pondérée**
avec deux séries **Train / Valid** pour chaque loss. Le **learning rate** est suivi
uniquement pour Train. La loss totale vaut `présence + LOCALIZATION_WEIGHT × localisation` ;
le poids vaut actuellement 1, donc les deux courbes de localisation sont identiques.
Les valeurs par lot et les moyennes de l'époque sont affichées ; les moyennes des
losses utilisent le nombre de positifs (borné à 1), comme leur normalisation.
Les moyennes Train et Valid sont séparées et remises à zéro à chaque époque.
Les diagnostics des tailles des boîtes, des positifs et des collisions sont
conservés dans `runs/dilated/training.log`, recréé à chaque lancement. Chaque ligne
indique l'époque, la phase Train ou Valid et le lot.

- `←` / `→` : choisir la métrique à tracer.
- `↑` / `↓` : changer la vue du graphique.
- `q`, puis `s` : arrêter après le lot en cours et sauvegarder les poids.

Les poids sont enregistrés dans `runs/dilated/model.bpk` après la phase Train de
chaque époque, avant la validation, et lors d'un arrêt pendant Train. Un arrêt
pendant Valid conserve donc les derniers poids entraînés. Chaque sauvegarde est
écrite dans `model.next.bpk`, puis remplace le checkpoint précédent une fois terminée.
Ces chemins sont relatifs
au répertoire de lancement, comme pour le serveur de visualisation. Si la sortie est redirigée vers
un fichier ou un pipe, Burn utilise son affichage texte et les diagnostics restent
imprimés. Le terminal est restauré à la fin de l'entraînement.

En cas de panique Rust, `runs/dilated/failure.log` conserve le message, la backtrace,
l'époque, la phase, le lot et les chemins des images en cours. Les erreurs ordinaires
sont aussi enregistrées. Le rapport est complété à chaque erreur, sans écraser les
précédentes. Un message indique son chemin après la fermeture de l'UI.
Le logger natif de Burn écrit aussi les messages internes (GPU, renderer, etc.)
dans `runs/dilated/runtime.log`.

Pour repartir de poids sauvegardés après six époques, depuis `face_lookup` :

```sh
cargo run --release -- --train --resume-weights runs/dilated/model.bpk --start-epoch 7
```

Les poids sont chargés avant d'ouvrir l'UI et de recréer `training.log`.
La numérotation des époques et le taux d'apprentissage reprennent au point indiqué.
**L'état d'Adam n'était pas sauvegardé : ses moments sont réinitialisés.** Il s'agit
donc d'une reprise des poids, pas d'une continuation strictement identique du calcul.
Sans ces options, `--train` démarre toujours un nouveau modèle à l'époque 1.

### Entraîner en plus haute résolution

Depuis `face_lookup`, pour un **nouveau modèle en 640×640** :

```sh
cargo run --release -- --train --image-size 640
```

Le réseau est entièrement convolutionnel : `Model::default` conserve la même
architecture et initialise de nouveaux poids. L'entrée passe à 640×640 et la grille
de détection à 80×80, au lieu de 320×320 et 40×40. Chaque dimension est doublée,
soit quatre fois plus de pixels. Le champ récepteur reste de 255 pixels de l'entrée.
Ce changement vise les petits visages ; son effet sur la qualité reste à mesurer.

Les poids sont enregistrés dans **`runs/dilated-640/model.bpk`**, et les journaux dans
ce même dossier. Le modèle 320 de `runs/dilated` est conservé. `--run-dir runs/essai-640`
permet de choisir un autre dossier. Un dossier contenant déjà `model.bpk` est refusé
pour un nouveau modèle ; pour continuer ce modèle, utiliser `--resume-weights`.
La résolution d'un dossier existant ne peut pas être changée.

En haute résolution, le lot par défaut est **1 image** pour limiter la mémoire GPU ;
`--batch-size 2` permet de l'augmenter. En 320, le lot par défaut reste de 4 images.
Les résolutions acceptées sont les multiples de 32 entre 320 et 1280. Le temps de
calcul et la mémoire augmentent avec la résolution ; le lot plus petit modifie aussi
le nombre de mises à jour par époque.

Pour initialiser un nouvel entraînement 640 à partir des poids déjà appris en 320 :

```sh
cargo run --release -- --train --image-size 640 --resume-weights runs/dilated/model.bpk
```

Cette commande écrit également dans `runs/dilated-640`, commence à l'époque 1
avec le calendrier de taux d'apprentissage habituel, et réinitialise Adam.
Choisir un autre `--run-dir` si ce dossier contient déjà un modèle.

Pour reprendre le modèle 640 après 10 époques terminées :

```sh
cargo run --release -- --train --resume-weights runs/dilated-640/model.bpk --start-epoch 11
```

La résolution est lue dans **`model.config.json`**, enregistré à côté des poids.
**Conserver ces deux fichiers ensemble** (si les poids sont renommés `essai.bpk`,
renommer aussi la configuration `essai.config.json`). Le viewer utilise automatiquement
cette résolution à la sélection du checkpoint. Les anciens fichiers `.bpk` sans
configuration restent interprétés en 320×320. Redémarrer le serveur après compilation
pour utiliser le nouveau chargement. Pour reprendre dans un dossier personnalisé,
passer de nouveau son `--run-dir`.

Chaque époque entraîne le modèle sur les **12 880 images de train**, puis évalue
les **3 226 images de val**, avec la taille de lot choisie, sans filtrage par nombre de visages.
La validation emploie `model.valid()` sur un appareil sans autodiff, sans calcul
de gradients, sans pas d'optimiseur et sans avancer le taux d'apprentissage.
Le modèle d'entraînement conserve ses poids et son mode d'entraînement.
Les réglages restent 100 époques, Adam et décroissance cosinus du taux d'apprentissage.
Le jeu `test` officiel n'est pas utilisé. Les losses de validation servent au suivi ;
elles ne remplacent pas les métriques officielles de détection WIDER FACE.

## Visualisation des visages

Ouvrir `http://127.0.0.1:3001/` après le lancement du serveur : la page HTML
utilise l'API pour afficher une grille paginée (12, 24 ou 48 images), avec accès
direct à une page. La case « Zones des visages » alterne entre l'original et
l'image annotée ; cliquer sur une image ouvre sa version en taille réelle.

`GET /api/train/{id}/annotated` renvoie un **PNG** avec les rectangles dessinés
côté serveur, en vert pour les annotations valides et orange pour celles dont
`invalid` est non nul. Les coordonnées restent celles de l'image originale ;
les rectangles vides sont ignorés et les rectangles débordants sont bornés à
l'image. La liste JSON fournit aussi `annotated_image_url`.

Le module `face_lookup/src/render.rs` contient `draw_face_boxes(&mut RgbImage,
&[FaceBox])`, réutilisable sans HTTP ni accès aux fichiers, et `annotated_png`
pour le décodage/rendu/encodage. Le serveur exécute ce traitement dans
`spawn_blocking`. Les fichiers originaux ne sont jamais modifiés.
