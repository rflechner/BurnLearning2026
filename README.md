# BurnLearning2026

## API REST du dataset d'entraînement

Depuis la racine du dépôt :

```sh
cargo run --manifest-path face_lookup/Cargo.toml
```

Le serveur écoute sur `http://127.0.0.1:3000`. Les annotations sont chargées une
seule fois au démarrage ; les JPEG sont lus à la demande, sans décodage ni modification.
Une annotation invalide empêche le démarrage avec une erreur de parsing.

### Parcourir les images

```sh
curl 'http://127.0.0.1:3000/api/train?page=1&page_size=20'
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
curl 'http://127.0.0.1:3000/api/train/0/image' -o image.jpg
```

Renvoie le JPEG original avec `Content-Type: image/jpeg`. Un identifiant inconnu
ou un fichier absent renvoie HTTP 404. Seuls les fichiers référencés par les
annotations et situés dans le dossier des images sont accessibles.

### Configuration

Variables d'environnement facultatives :

- `BIND_ADDR` : adresse d'écoute (défaut `127.0.0.1:3000`).
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
