# handwritten-gen — comprendre comment une machine génère une image

Ce projet est un terrain d'apprentissage en Rust avec Burn. On part de presque
zéro pour comprendre comment un programme peut apprendre à dessiner des chiffres
manuscrits, puis en créer de nouveaux.

L'idée qui va nous guider : **une image est une liste de nombres. Un générateur
apprend à calculer des listes de nombres qui ressemblent à des images du dataset.**

Pas besoin de maîtriser toutes les formules dès le début. Avance dans l'ordre,
fais les petites expériences et regarde les images obtenues.

**État du code :** `src/main.rs` affiche actuellement `Hello, world!`.
`Cargo.toml` déclare Burn `0.22.1` et `image`. Les étapes ci-dessous sont à
implémenter. Le premier exercice contient du Rust à essayer ; les blocs marqués
« pseudo-code » expliquent le fonctionnement et ne sont pas des API Burn.

## Le voyage que l'on va faire

| Étape | Ce que l'on découvre | Ce que l'on doit obtenir |
| --- | --- | --- |
| 1 | Une image est faite de nombres | Un PNG dessiné par du code |
| 2 | MNIST fournit des exemples | Une grille de vrais chiffres |
| 3 | Un réseau ajuste des paramètres | Une erreur qui diminue |
| 4 | Compresser puis reconstruire | Un chiffre reconstruit |
| 5 | Organiser les codes pour en tirer au hasard | Un VAE entraîné |
| 6 | Générer sans image modèle | Une grille de nouveaux chiffres |
| 7 | Donner une consigne au générateur | Plusieurs variantes du chiffre choisi |

Notre premier objectif concerne les **chiffres isolés**, pas encore les mots.
[MNIST](https://www.tensorflow.org/datasets/catalog/mnist) contient des images
28 × 28 en niveaux de gris des chiffres 0 à 9 : 60 000 pour l'entraînement et
10 000 pour le test. Il ne contient ni lettres ni phrases cursives.

## 1. Avant l'IA : fabriquer une image avec des nombres

### Un pixel, c'est quoi pour notre programme ?

Dans une image en niveaux de gris, un pixel est une luminosité :

- `0` : noir ;
- `255` : blanc ;
- `128` : un gris intermédiaire.

Une image de 28 × 28 pixels contient donc `28 * 28 = 784` nombres.
Dans MNIST, les traits sont clairs sur un fond sombre.

Voici une toute petite image imaginaire, de 5 × 5 pixels :

```text
  0    0  255    0    0
  0  255  255    0    0
  0    0  255    0    0
  0    0  255    0    0
  0  255  255  255    0
```

Si tu remplaces mentalement les `255` par des carrés blancs, un « 1 » apparaît.
Pour produire une image, notre futur réseau devra simplement produire les
bonnes luminosités aux bons endroits.

### À essayer : dessiner sans apprendre

Pour cet exercice, remplace temporairement le contenu de `src/main.rs` par :

```rust
use image::{GrayImage, Luma};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Une nouvelle image en niveaux de gris, initialement noire.
    let mut img = GrayImage::new(28, 28);

    // x = colonne, y = ligne. On trace une barre verticale blanche.
    for y in 5..23 {
        for x in 12..16 {
            img.put_pixel(x, y, Luma([255u8]));
        }
    }

    img.save("premier-chiffre.png")?;
    println!("Image créée : premier-chiffre.png");
    Ok(())
}
```

Depuis le dossier `handwritten-gen`, lance :

```sh
cargo run
```

Le PNG sera créé dans ce dossier. Ouvre-le et zoome : 28 pixels, c'est petit !
La compilation utilise les dépendances du manifeste actuel, même si ce premier
exercice n'utilise pas encore Burn.

**Expériences :** remplace `255` par `100`, déplace la barre, puis ajoute une
barre horizontale. Tu modifies l'image en modifiant ses nombres.

**À comprendre avant la suite :** ici, c'est toi qui as choisi où mettre les
pixels. Avec l'apprentissage, ce seront les paramètres du réseau qui
permettront de calculer ces positions et ces intensités.

## 2. Donner des exemples : découvrir MNIST

Un **dataset**, c'est un ensemble d'exemples. Dans MNIST, chaque exemple contient
une image et un **label**, c'est-à-dire le chiffre représenté.

```text
image de sept manuscrit → label 7
image de deux manuscrit → label 2
```

Pour reconnaître un chiffre, on apprendrait `image → label`.
Pour générer, on veut aller vers `quelques nombres → image`.
On commencera sans utiliser les labels, puis on les ajoutera pour choisir le
chiffre à produire.

### Préparer les données, pas à pas

1. Télécharger la version Parquet de MNIST sur Hugging Face, puis lire une
   première image avec le parcours détaillé ci-dessous. Le format historique
   de MNIST est IDX, mais nous allons partir de cette distribution Parquet.
2. Afficher les dimensions et les valeurs minimale et maximale d'une image.
3. Exporter quelques images en PNG avec leurs labels dans les noms de fichiers.
   Vérifier qu'un fichier annoncé comme « 7 » contient bien un sept.
4. Convertir chaque pixel en nombre flottant entre `0.0` et `1.0` :
   `pixel_normalise = pixel as f32 / 255.0`.
5. Transformer chaque grille en une liste de 784 valeurs, en conservant toujours
   le même ordre : ligne après ligne.

Cette conversion s'appelle la **normalisation**. Elle donne une échelle simple
pour les calculs du réseau. `0` reste noir, `1` devient blanc, et `0.5` est gris.
On garde les gris : inutile de transformer tous les pixels en noir ou blanc.

### Notre source : MNIST sur Hugging Face

Le dépôt [ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist/tree/main/mnist)
distribue MNIST sous forme de deux fichiers Parquet :

| Fichier | Contenu |
| --- | --- |
| `train-00000-of-00001.parquet` | 60 000 exemples d'entraînement |
| `test-00000-of-00001.parquet` | 10 000 exemples de test |

La [fiche du dataset](https://huggingface.co/datasets/ylecun/mnist/blob/main/README.md)
attribue MNIST à Yann LeCun, Corinna Cortes et Chris Burges. Il s'agit ici d'une
distribution convertie en Parquet sur Hugging Face, pas du format original.
Le nom `00000-of-00001` indique le premier et unique morceau du split : aucun
autre morceau n'est à chercher pour chacun de ces deux fichiers.

### Parquet, c'est une image ?

Non : c'est un format de stockage de tableaux, organisé par colonnes. Imagine
un tableau dont chaque ligne représente un exemple et dont les colonnes sont
`image` et `label`. La colonne image contient les données permettant de retrouver
l'image ; la colonne label contient le chiffre attendu.

La représentation Hugging Face d'une image peut comporter des champs `bytes`
(contenu du fichier image) et `path` (chemin). Il faut inspecter le schéma du
Parquet et une première ligne pour vérifier les champs présents et non nuls,
plutôt que supposer qu'il contient directement 784 nombres.
Voir la [documentation des images Hugging Face](https://huggingface.co/docs/datasets/about_dataset_features#image-feature).

Il y a donc **deux lectures différentes** :

```text
Fichier Parquet
    → lecteur Parquet : extraire une ligne, son label et les octets de l'image
    → décodeur d'image : transformer ces octets en pixels
    → normalisation : convertir les pixels en flottants entre 0 et 1
    → Burn : regrouper ces nombres dans un tenseur
```

Les octets d'un fichier PNG ne sont pas ses pixels bruts : ils contiennent
aussi un en-tête et des données compressées. Il ne faut donc pas diviser les
octets du fichier par 255 avant de l'avoir décodé !

### Télécharger les deux fichiers

Depuis le dossier `handwritten-gen`, exécuter :

```sh
mkdir -p data/mnist
curl --fail --location \
  'https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/train-00000-of-00001.parquet' \
  --output data/mnist/train.parquet
curl --fail --location \
  'https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet' \
  --output data/mnist/test.parquet
```

`resolve` donne accès au fichier lui-même ; `tree` affiche la page de navigation.
L'option `--location` suit les redirections du téléchargement et `--fail`
signale les erreurs HTTP. Ces commandes sont à lancer pour l'exercice : le
tutoriel ne télécharge pas automatiquement les données.

### Premier exercice de lecture en Rust

La crate `image` sait décoder les images, mais pas ouvrir un tableau Parquet.
Pour cette étape, ajouter un lecteur Parquet au projet :

```sh
cargo add parquet
```

La crate Apache fournit notamment
[`SerializedFileReader`](https://docs.rs/parquet/latest/parquet/file/serialized_reader/struct.SerializedFileReader.html).
Commencer par ouvrir `data/mnist/train.parquet`, afficher son schéma et lire une
seule ligne. Les accès aux champs dépendront de la structure inspectée.

```text
Pseudo-code — à traduire en Rust après inspection du schéma :

ouvrir le fichier Parquet
afficher son schéma et son nombre de lignes
lire la première ligne
récupérer le label
récupérer les octets encodés dans le champ image
décoder ces octets avec la crate image
vérifier que les dimensions sont 28 × 28
exporter cette image en PNG avec le label dans son nom
```

Une fois les octets extraits dans une variable `bytes`, voici la partie décodage
en Rust, à placer dans une fonction qui retourne un `Result` :

```rust
// bytes : &[u8], contenu encodé d'UNE image, extrait du Parquet.
let img = image::load_from_memory(bytes)?.to_luma8();
assert_eq!(img.dimensions(), (28, 28));
img.save("premiere-image-mnist.png")?;

let pixels: Vec<f32> = img.as_raw()
    .iter()
    .map(|&pixel| pixel as f32 / 255.0)
    .collect();
assert_eq!(pixels.len(), 784);
```

`load_from_memory` décode un fichier image stocké en mémoire ; `to_luma8`
produit une `GrayImage`. Tu retrouves donc la famille de types que tu as déjà
manipulée avec `RgbImage`, avec un seul canal au lieu de trois.
Voir la [documentation du décodeur](https://docs.rs/image/latest/image/fn.load_from_memory.html).

Si le champ `bytes` est absent ou nul, examiner `path` et la manière dont les
images sont distribuées avant de poursuivre. Ne pas remplacer une image absente
par des pixels noirs : cela cacherait un problème de chargement.

**Objectif de cette séance :** ouvrir un seul exemple, afficher son label et
voir son image. Ensuite seulement, exporter dix exemples, puis constituer les
lots. Aucun réseau de neurones n'est nécessaire pour cette étape.

### Le mot « tenseur » sans mystère

Un tenseur est un tableau de nombres avec une forme :

| Contenu | Forme possible |
| --- | --- |
| Une image aplatie | `[784]` |
| Un lot de 128 images aplaties | `[128, 784]` |
| Le même lot sous forme d'images avec un canal de gris | `[128, 1, 28, 28]` |

Un **batch**, ou lot, permet de traiter plusieurs images à la fois. Dans notre
premier modèle, on utilisera la forme `[taille_du_lot, 784]`.

### Pourquoi séparer les exemples ?

- **Entraînement :** les exemples qui servent à modifier les paramètres.
- **Validation :** des exemples à part pour guider nos réglages.
- **Test :** des exemples gardés pour l'évaluation finale.

On peut réserver 5 000 images parmi les 60 000 d'entraînement pour la validation.
Il reste alors 55 000 images pour apprendre. Les 10 000 images de test restent
à part. Fixer une graine pour rendre cette séparation reproductible.

**Petit exercice :** prends une image, aplatis-la en 784 valeurs, remets-la en
28 × 28 et exporte-la. Elle doit être identique. Si elle est tournée ou déformée,
corrige l'ordre des pixels avant de continuer.

## 3. Comment un réseau apprend-il quelque chose ?

Un réseau est une fonction avec des nombres réglables, appelés **paramètres**
ou **poids**. Au départ, ils sont généralement initialisés aléatoirement.

Un calcul élémentaire ressemble à ceci :

```text
sortie = poids_1 * entrée_1 + poids_2 * entrée_2 + biais
```

Le biais est un autre nombre réglable, ajouté à la somme. Une couche `Linear`
fait beaucoup de calculs de ce type. `Linear(784, 256)` transforme 784 nombres
en 256 nombres. Elle possède `784 * 256` poids et 256 biais.

Entre les couches, on ajoute une fonction non linéaire. Par exemple,
`ReLU(x) = max(0, x)` remplace les nombres négatifs par zéro. Sans ces fonctions,
empiler des couches linéaires resterait équivalent à une seule transformation
linéaire, trop limitée pour ce que l'on cherche à apprendre.

### La boucle d'apprentissage

1. Le réseau produit une sortie : c'est le **passage avant**.
2. On mesure son erreur avec une **fonction de perte** (*loss*).
3. On calcule comment chaque paramètre influence cette erreur : ce sont les
   **gradients**, obtenus par rétropropagation.
4. Un **optimiseur** utilise les gradients pour modifier les paramètres.
5. On recommence avec d'autres exemples.

Exemple miniature : on voulait une luminosité `0.8`, le modèle prédit `0.3`.
Une erreur quadratique vaut `(0.3 - 0.8)² = 0.25`. Si la prédiction passe à
`0.6`, elle vaut `0.04` : elle a diminué.

Le gradient ne dit pas « dessine une boucle de neuf ». Il indique comment
modifier les nombres du réseau pour réduire l'erreur. À force de traiter des
exemples, ces modifications peuvent permettre d'apprendre des formes utiles.

Le **taux d'apprentissage** règle l'ampleur des mises à jour. Trop grand, il peut
rendre l'apprentissage instable ; trop petit, la progression peut être lente.
Une **époque** correspond à un passage sur tout le jeu d'entraînement.

### Et Burn, dans tout ça ?

Burn fournit les tenseurs, les couches, les gradients automatiques et les
optimiseurs. Le **backend** est le moteur qui exécute les calculs sur le matériel.
Le manifeste déclare plusieurs backends : en choisir un compatible avec la
machine pour commencer.

Avant le modèle complet, faire fonctionner un petit calcul sur ce backend et
vérifier que l'on peut calculer des gradients. Le
[guide Burn](https://burn.dev/books/burn/basic-workflow/training.html) illustre
l'entraînement sur MNIST avec un classifieur. Les principes sont utiles ici,
mais notre sortie sera une image et notre perte sera différente. Vérifier les
API contre la version `0.22.1` du projet : le guide en ligne peut évoluer.

## 4. Première marche : reconstruire une image

Avant d'inventer un chiffre, apprenons à en reconstruire un. On utilise un
**auto-encodeur**, composé de deux parties :

```text
Image originale       Petit code         Image reconstruite
 784 nombres     →    16 nombres    →       784 nombres
                  encodeur          décodeur
```

L'**encodeur** transforme l'image en un petit code. Le **décodeur** transforme ce
code en une image. On appelle ce code la **représentation latente** : des nombres
intermédiaires que le réseau apprend à utiliser.

Pourquoi passer par seulement 16 nombres ? Cela oblige le modèle à condenser
l'information. Il ne peut pas simplement transporter les 784 pixels tels quels.
Ce n'est pas une compression sans perte : certains détails peuvent disparaître.

Architecture d'exercice :

```text
Encodeur : 784 → Linear → 256 → ReLU → Linear → 16
Décodeur :  16 → Linear → 256 → ReLU → Linear → 784 → sigmoid
```

La **sigmoid** ramène chaque sortie entre 0 et 1 pour représenter une luminosité.
Par exemple, elle transforme `0` en `0.5`, un grand nombre positif en une valeur
proche de `1`, et un grand nombre négatif en une valeur proche de `0`.

### À implémenter

Commencer sur un petit sous-ensemble, par exemple 256 images. Donner l'image en
entrée et utiliser **cette même image comme cible**. Pour cet exercice, mesurer
la moyenne des erreurs quadratiques entre les pixels originaux et reconstruits.

```text
Pseudo-code :

pour chaque lot d'images :
    code = encodeur(images)
    reconstruction = decodeur(code)
    perte = moyenne((reconstruction - images)²)
    gradients = calculer_gradients(perte)
    mettre_a_jour_les_parametres(gradients)
```

La mise à jour doit concerner l'encodeur et le décodeur. Ne pas accumuler
involontairement les gradients entre les lots.

**À observer :** exporter côte à côte les images originales et reconstruites.
Sur ce petit ensemble, le réseau doit progressivement mieux reconstruire les
chiffres. Ce test vérifie le fonctionnement, pas la capacité à généraliser.

### Pourquoi cela ne suffit-il pas encore pour générer ?

On pourrait tirer 16 nombres au hasard et les donner au décodeur. Mais rien
n'oblige cet auto-encodeur à avoir appris à décoder ces endroits-là.

Imagine une carte où seules quelques zones ont été explorées : choisir une
position au hasard peut tomber loin des zones connues. Une bonne reconstruction
ne garantit donc pas une bonne génération à partir de nombres aléatoires.

## 5. Le VAE : apprendre un espace dans lequel on peut tirer au hasard

C'est le rôle de l'**auto-encodeur variationnel**, ou **VAE**. Il reprend
l'encodeur et le décodeur, mais change la façon de construire le petit code.

L'objectif est double : reconstruire les images et rapprocher les distributions
de leurs codes d'une distribution connue, dans laquelle on saura tirer de
nouveaux nombres. Ce rapprochement favorise la génération ; il ne garantit pas
que tous les tirages produiront de beaux chiffres.

### Un code devient une petite zone de possibilités

Au lieu de sortir directement 16 nombres, l'encodeur produit :

- `mu` : 16 nombres qui donnent le centre de la distribution ;
- `log_var` : 16 nombres qui décrivent son étalement.

`mu` se lit « mu » et désigne une moyenne. `log_var` est le logarithme de la
variance. On utilise cette représentation pour pouvoir calculer ensuite une
variance positive, même si la sortie du réseau est négative.

On tire alors un code près de ce centre :

```text
sigma = exp(0.5 * log_var)
z = mu + sigma * epsilon
```

Chaque composante de `epsilon` est tirée dans une **loi normale standard** :
une distribution en cloche, centrée sur zéro, d'écart-type 1. On la note souvent
`N(0, 1)`. Pour un vecteur de composantes indépendantes, on écrit `N(0, I)`.

Exemple avec une seule dimension : si `mu = 0.4`, `sigma = 0.2` et le tirage
`epsilon = -1`, alors `z = 0.2`. Un autre tirage donne un autre code près du
même centre. Dans notre modèle, ce calcul est fait sur 16 dimensions.

Cette formule est la **reparamétrisation** : le hasard vient de `epsilon`,
et les opérations sur `mu` et `sigma` permettent de calculer les gradients.

### Les deux consignes données au modèle

**Consigne 1 : reconstruis correctement.** La perte de reconstruction pénalise
les différences entre l'image d'entrée et l'image produite.

**Consigne 2 : garde des distributions latentes proches de la normale standard.**
Un deuxième terme, appelé **divergence KL**, mesure cet écart. Il permet de
rapprocher les codes utilisés pendant l'entraînement des codes que l'on tirera
au hasard pendant la génération.

Sans la première consigne, le code pourrait cesser de représenter l'image.
Sans la seconde, on retrouve le problème des zones inconnues de l'auto-encodeur.

```text
perte_totale = perte_reconstruction + beta * perte_KL
```

`beta` règle l'importance de la deuxième consigne. Commencer avec `beta = 1`.
Un poids trop fort peut pousser le décodeur à ignorer le code ; trop faible,
il peut laisser un espace latent mal adapté à la génération.

<details>
<summary>Pour l'implémentation : les formules et les dimensions</summary>

Architecture proposée, avec une dimension latente de 16 :

```text
images [B, 784]
  → Linear(784, 256) + ReLU
  → deux couches distinctes Linear(256, 16)
  → mu [B, 16] et log_var [B, 16]
  → z = mu + exp(0.5 * log_var) * epsilon
  → Linear(16, 256) + ReLU + Linear(256, 784)
  → logits [B, 784]
```

`B` est la taille du lot. Les **logits** sont les sorties brutes avant sigmoid.
Pour produire des pixels, on leur applique la sigmoid.

Pour la reconstruction du VAE, on peut utiliser une entropie croisée binaire
(BCE) avec logits, courante sur MNIST, en gardant les cibles dans `[0, 1]`.
Intuitivement, elle pénalise les luminosités prédites qui s'écartent des cibles.
Avec les niveaux de gris, c'est une convention pratique de reconstruction.

Une expression stable par pixel, pour un logit `a` et une cible `x`, est :

```text
BCE(a, x) = max(a, 0) - a * x + ln(1 + exp(-abs(a)))
```

Préférer une opération stable fournie par la bibliothèque lorsqu'elle existe.
Une fonction « BCE avec logits » attend les sorties brutes : ne pas lui donner
une sortie déjà passée dans la sigmoid.

La KL par image est :

```text
KL = -0.5 * somme(1 + log_var - mu² - exp(log_var))
```

Sommer la reconstruction sur les 784 pixels et la KL sur les 16 dimensions,
puis moyenner chacun des deux termes sur les images du lot. Changer ces sommes
en moyennes change leur importance relative : conserver une convention claire.

Le fondement du modèle est présenté dans
[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
La lecture de l'article n'est pas un prérequis pour faire les exercices.

</details>

### Entraîner progressivement

Point de départ proposé : optimiseur Adam, taux d'apprentissage `0.001`, lots de
128 images, dimension latente 16. Ce sont des réglages d'essai, pas une recette
garantissant un résultat.

1. Faire quelques mises à jour sur un petit ensemble. Vérifier les dimensions,
   les gradients et l'absence de valeurs invalides (`NaN`).
2. Passer au jeu d'entraînement complet. Mélanger les images à chaque époque.
3. Mesurer séparément reconstruction et KL sur l'entraînement et la validation.
   En validation, ne pas mettre à jour les poids ni suivre les gradients.
4. Après chaque époque, exporter des reconstructions **et** des images créées
   depuis des codes aléatoires. Garder les mêmes codes pour comparer les époques.
5. Essayer 10 à 30 époques, puis décider à partir des courbes et des images.
6. Enregistrer un **checkpoint** : les poids du modèle, accompagnés de sa
   configuration. Conserver celui choisi selon une règle de validation fixée.

Pour comparer les reconstructions, on peut décoder `mu` sans tirage aléatoire.
Pour comparer une perte VAE complète, conserver le même protocole
d'échantillonnage entre les évaluations.

**Petit exercice :** regarder séparément la reconstruction et la KL. Si seule
la perte totale est affichée, il est difficile de savoir quelle consigne le
modèle apprend à satisfaire.

## 6. Le moment clé : créer une image qui n'était pas fournie en entrée

Après l'entraînement, on peut mettre l'encodeur de côté. Le décodeur a appris
une fonction qui transforme un code en pixels.

```text
Pendant l'entraînement :
image MNIST → encodeur → code z → décodeur → reconstruction
       └──────────────── comparaison ───────────────┘
                + régularisation du code

Pendant la génération :
16 nombres aléatoires → décodeur entraîné → 784 luminosités → PNG
```

**On ne prend pas une image MNIST pour la modifier.** On tire un nouveau code
et le décodeur calcule une image avec ses paramètres appris. Les exemples ont
influencé ces paramètres pendant l'entraînement ; ils ne sont plus nécessaires
pour effectuer ce calcul. Cela n'exclut pas une mémorisation de certains exemples,
que l'on pourra examiner ensuite.

### D'où vient la forme du chiffre ?

Le hasard fournit un point de départ, mais il ne connaît rien aux chiffres.
C'est le décodeur entraîné qui transforme ce point de départ en une structure
visuelle. Avec un décodeur non entraîné, les mêmes codes ne donneraient pas des
chiffres appris.

Et les 16 nombres ne sont pas nécessairement des commandes lisibles comme
« inclinaison », « épaisseur » ou « chiffre ». Le modèle construit sa propre
représentation, souvent répartie entre plusieurs dimensions.

### À implémenter

```text
Pseudo-code :

decodeur = charger_les_poids_et_la_configuration()
z = tirer_normale_standard(forme = [64, 16])
logits = decodeur(z)
pixels = sigmoid(logits)
images = remettre_en_forme(pixels, [64, 28, 28])
exporter_une_grille(images)
```

Pour chaque pixel, convertir la luminosité en octet en multipliant par 255,
en arrondissant et en bornant le résultat entre 0 et 255.

**Vérification importante :** arrêter le programme, puis lancer la génération
dans un nouveau processus en rechargeant le checkpoint. Cela doit fonctionner
sans charger MNIST ni utiliser l'encodeur.

### Trois expériences pour comprendre la génération

**1. Rejouer le même code.** Avec les mêmes poids, le même `z` et un décodeur en
mode évaluation sans opération aléatoire, on doit retrouver la même image,
à d'éventuelles différences numériques près. Le hasard est dans le tirage du
code, pas dans une imagination qui changerait à chaque appel.

**2. Modifier une coordonnée.** Garder 15 coordonnées fixes et faire varier la
seizième par petites étapes. Observer ce qui change. Une coordonnée peut
modifier plusieurs aspects à la fois, ou avoir peu d'effet visible.

**3. Voyager entre deux codes.** Choisir `z_A` et `z_B`, puis calculer :

```text
z = (1 - t) * z_A + t * z_B
avec t = 0, 0.1, 0.2, ..., 1
```

Décoder chaque point et placer les images côte à côte. C'est une
**interpolation** : on explore ce que le réseau produit entre deux positions
latentes. Les étapes intermédiaires peuvent être ambiguës ; elles ne sont pas
forcément des chiffres parfaits.

## 7. Comment demander précisément un « 7 » ?

Notre premier VAE génère des chiffres sans permettre de choisir leur identité.
Pour donner cette consigne, on construit un **VAE conditionnel**, ou **CVAE**.
« Conditionnel » signifie simplement que le modèle reçoit une information
supplémentaire pour guider sa sortie.

On représente le label avec dix cases, une par chiffre. Une seule vaut 1 :

```text
chiffres : 0 1 2 3 4 5 6 7 8 9
label 7 : [0,0,0,0,0,0,0,1,0,0]
```

C'est un encodage **one-hot**. On ajoute ces dix nombres aux entrées :

```text
Encodeur : image [784] + label [10] → mu et log_var
Décodeur : code z [16] + label [10] → image [784]
```

Il faut réentraîner ce modèle avec les labels. Ajouter un label à un décodeur
qui n'a jamais appris à s'en servir ne suffit pas.

À la génération, le label reste `7`, mais on change `z` pour obtenir plusieurs
variantes. Le label indique le chiffre souhaité ; le code peut représenter
les variations restantes, sans garantir des contrôles de style indépendants.

**Exercice :** créer une grille de dix lignes, une par chiffre, et huit colonnes
avec des codes différents. Vérifier que les lignes respectent la consigne et
que les images ne sont pas toutes identiques.

On retrouve ici une idée générale de la génération guidée : fournir une
condition au modèle. Une consigne textuelle complète demanderait d'autres
représentations, d'autres données et un modèle adapté.

## 8. Savoir si l'on progresse et quoi regarder en cas de problème

Une perte plus basse est utile, mais ne suffit pas : regarder aussi les images.
Ne pas sélectionner uniquement les meilleurs résultats pour juger le modèle.

| Observation | Première vérification |
| --- | --- |
| Images toutes noires ou blanches | Normalisation, export et sigmoid |
| Images tournées ou mélangées | Ordre des pixels et formes des tenseurs |
| Perte qui devient `NaN` | Calculs stables, taux d'apprentissage, valeurs de `log_var` |
| Reconstruction correcte, génération mauvaise | Tirage normal de `z`, KL et équilibre des pertes |
| Sorties toutes similaires | Effet de `z`, poids de la KL, diversité des données |
| Chiffres reconnaissables mais flous | Limite possible du modèle ; essayer ensuite des couches convolutionnelles |
| CVAE qui ignore le chiffre demandé | Labels fournis correctement à l'entraînement et à la génération |

Un classifieur indépendant peut aider à mesurer si les chiffres conditionnels
sont reconnus, à condition de vérifier d'abord sa précision sur le test MNIST.
Ce score ne remplace pas l'inspection visuelle.

Comparer aussi quelques générations à leurs images d'entraînement les plus
proches pour repérer des copies évidentes. Cette inspection ne prouve pas à
elle seule l'absence de mémorisation.

Conserver les paramètres, les graines, les versions et les checkpoints.
Une **graine aléatoire** permet de reproduire une suite de tirages ; elle ne
supprime pas toutes les différences possibles entre matériels ou backends.

## 9. Et ensuite : lettres, mots et autres générateurs

Générer séparément les quatre chiffres de `2026` puis assembler les PNG est
une première extension amusante. Cela ne crée pas une écriture liée, ni un
style d'auteur cohérent entre les chiffres.

Pour les lettres, il faudra des exemples de lettres. Pour les mots et les
phrases, il faudra notamment des images de texte avec leurs transcriptions,
un modèle conditionné par ce texte et une gestion des longueurs variables.
Pour un style cohérent, il faudra des données ou références de style adaptées.
Pour reproduire des gestes de stylet, il faudra des trajectoires : une image
MNIST ne dit pas dans quel ordre les traits ont été dessinés.

Le VAE est notre première porte d'entrée vers la génération. Il permet de
comprendre les pixels, l'apprentissage, les codes latents et le conditionnement.
On pourra ensuite étudier d'autres familles de modèles, dont la diffusion.
Notre VAE ne retire pas progressivement le bruit d'une image : il transforme
un petit vecteur aléatoire en pixels avec son décodeur.

## Comment organiser le code au fil des exercices

Ne crée pas toute l'architecture avant d'avoir exporté ta première image.
Extrais les modules à mesure que les étapes fonctionnent :

```text
src/
  main.rs        # lancer les exercices, puis entraîner ou générer
  dataset.rs     # charger MNIST et préparer les lots
  model.rs       # encodeur et décodeur
  training.rs    # calculer les pertes et mettre à jour les poids
  generate.rs    # recharger les poids et exporter les images
```

Seul `main.rs` existe actuellement. Il faudra aussi prévoir des dossiers pour
les données, les checkpoints et les images, en excluant les fichiers volumineux
du suivi Git.

### Ton premier objectif concret

Commence par **l'exercice 1**, puis affiche de vraies images MNIST avec
**l'étape 2**. Tu auras compris ce qui entre et sort du futur réseau. Ensuite,
fais fonctionner la reconstruction avant de chercher à générer depuis du bruit.

Le projet aura atteint son premier objectif complet lorsqu'un programme pourra
recharger des poids et produire une grille de nouveaux chiffres reconnaissables,
sans image d'entrée.
