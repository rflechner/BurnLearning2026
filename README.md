# BurnLearning2026

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
activé. Les annotations seules ne suffisent pas : l'archive `WIDER_train.zip` doit
être extraite pour obtenir
`face_lookup/data/extracted/db171f1b7fedf4d3453e81297ff02f9915356d19/WIDER_train/images`.
Si les photos sont ailleurs, définir `IMAGES_ROOT` sur leur dossier `WIDER_train/images`
dans les variables d'environnement de la configuration. L'entraînement vérifie les
fichiers sélectionnés avant d'initialiser le GPU et d'ouvrir l'interface.

Le tableau de bord natif de Burn affiche la progression des époques et les courbes
de **loss totale**, **présence**, **localisation brute**, **localisation pondérée**
et **learning rate**. La loss totale vaut `présence + LOCALIZATION_WEIGHT × localisation` ;
le poids vaut actuellement 1, donc les deux courbes de localisation sont identiques.
Les valeurs par lot et les moyennes de l'époque sont affichées ; les moyennes des
losses utilisent le nombre de positifs (borné à 1), comme leur normalisation.
Les diagnostics des tailles des boîtes, des positifs et des collisions sont
conservés dans `runs/dilated/training.log`, recréé à chaque lancement.

- `←` / `→` : choisir la métrique à tracer.
- `↑` / `↓` : changer la vue du graphique.
- `q`, puis `s` : arrêter après le lot en cours et sauvegarder les poids.

Les poids sont enregistrés dans `runs/dilated/model.bpk` à chaque époque et lors
d'un arrêt demandé par l'interface. Ces chemins sont relatifs au répertoire de
lancement, comme pour le serveur de visualisation. Si la sortie est redirigée vers
un fichier ou un pipe, Burn utilise son affichage texte et les diagnostics restent
imprimés. Le terminal est restauré à la fin de l'entraînement.

L'entraînement conserve ses paramètres actuels : les 100 premières images ayant
au plus 20 annotations, lots de 4, 100 époques, Adam et décroissance cosinus du taux
d'apprentissage. Les métriques affichées concernent uniquement l'entraînement ;
aucun jeu de validation n'est encore configuré.

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
