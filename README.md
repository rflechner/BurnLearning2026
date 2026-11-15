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
