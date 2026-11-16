use std::fs::File;
use std::io::{self, BufRead, BufReader, Lines};
use std::iter::Peekable;
use std::path::{Component, Path, PathBuf};

/// One face annotation line: `x y w h blur expression illumination invalid occlusion pose`.
/// All original attributes are kept so the filtering policy can change later.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FaceBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub blur: u32,
    pub expression: u32,
    pub illumination: u32,
    pub invalid: u32,
    pub occlusion: u32,
    pub pose: u32,
}

impl FaceBox {
    pub fn is_degenerate(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// One annotated image: its relative path and its face boxes.
#[derive(Debug)]
pub struct AnnotatedImage {
    pub image_file_path: String,
    pub faces: Vec<FaceBox>,
}

/// Streams a WIDER FACE annotation file (`wider_face_*_bbx_gt.txt`) block by block:
/// image path line, face count line, then one 10-integer line per face.
///
/// Stays lazy (the Rust equivalent of C# `yield return`): a block is read and
/// parsed only when the iterator advances. Malformed blocks are yielded as `Err`.
pub struct WiderAnnotations<R: BufRead> {
    lines: Peekable<Lines<R>>,
    line_no: usize,
}

pub(crate) const ANNOTATIONS_PATH: &str = "data/extracted/db171f1b7fedf4d3453e81297ff02f9915356d19/wider_face_split/wider_face_train_bbx_gt.txt";
pub(crate) const VALID_ANNOTATIONS_PATH: &str = "data/extracted/db171f1b7fedf4d3453e81297ff02f9915356d19/wider_face_split/wider_face_val_bbx_gt.txt";

/// Annotation paths are relative to `WIDER_train/images/`, next to `wider_face_split/`.
pub fn images_root() -> PathBuf {
    Path::new(ANNOTATIONS_PATH)
        .parent()
        .and_then(|p| p.parent())
        .map(|extracted| extracted.join("WIDER_train/images"))
        .expect("annotations path has two parent directories")
}

pub fn validation_images_root() -> PathBuf {
    Path::new(VALID_ANNOTATIONS_PATH)
        .parent()
        .and_then(|p| p.parent())
        .map(|extracted| extracted.join("WIDER_val/images"))
        .expect("validation annotations path has two parent directories")
}

/// Opens an annotation file and returns its lazy block iterator.
pub fn load_annotations(path: impl AsRef<Path>) -> io::Result<WiderAnnotations<BufReader<File>>> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "Impossible d'ouvrir les annotations {} : {error}",
                path.display()
            ),
        )
    })?;
    Ok(WiderAnnotations::new(BufReader::new(file)))
}

/// Vérifie toutes les images d'entraînement avant l'initialisation du GPU et de l'UI.
pub fn load_training_samples(
    annotations_path: &Path,
    images_root: &Path,
) -> io::Result<Vec<AnnotatedImage>> {
    load_samples(
        annotations_path,
        images_root,
        "entraînement",
        "WIDER_train",
        "IMAGES_ROOT",
    )
}

pub fn load_validation_samples(
    annotations_path: &Path,
    images_root: &Path,
) -> io::Result<Vec<AnnotatedImage>> {
    load_samples(
        annotations_path,
        images_root,
        "validation",
        "WIDER_val",
        "VALID_IMAGES_ROOT",
    )
}

fn load_samples(
    annotations_path: &Path,
    images_root: &Path,
    split: &str,
    archive: &str,
    root_variable: &str,
) -> io::Result<Vec<AnnotatedImage>> {
    let samples = load_annotations(annotations_path)?.collect::<io::Result<Vec<_>>>()?;
    if samples.is_empty() {
        return Err(invalid_data(format!("Aucune image de {split}")));
    }

    let metadata = std::fs::metadata(images_root).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "Dossier des images de {split} inaccessible : {}. Extraire {archive}.zip dans le dossier du dataset ou définir {root_variable} vers {archive}/images. Détail : {error}",
                images_root.display(),
            ),
        )
    })?;
    if !metadata.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotADirectory,
            format!(
                "{root_variable} doit être un dossier : {}",
                images_root.display()
            ),
        ));
    }
    for sample in &samples {
        let relative = Path::new(&sample.image_file_path);
        if relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
        {
            return Err(invalid_data(format!(
                "Chemin image invalide : {}",
                relative.display()
            )));
        }
        let path = images_root.join(relative);
        let metadata = std::fs::metadata(&path).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("Image de {split} inaccessible : {}. Vérifier l'extraction de {archive}.zip et {root_variable}. Détail : {error}", path.display()),
            )
        })?;
        if !metadata.is_file() {
            return Err(invalid_data(format!(
                "L'image n'est pas un fichier : {}",
                path.display()
            )));
        }
    }
    Ok(samples)
}

/// Streams `data/train.train` in the WIDER block format.
pub fn load_train_data() -> io::Result<WiderAnnotations<BufReader<File>>> {
    load_annotations(ANNOTATIONS_PATH)
}

impl<R: BufRead> WiderAnnotations<R> {
    pub fn new(reader: R) -> Self {
        Self {
            lines: reader.lines().peekable(),
            line_no: 0,
        }
    }

    fn read_line(&mut self) -> Option<io::Result<String>> {
        let line = self.lines.next()?;
        self.line_no += 1;
        Some(line)
    }
}

impl<R: BufRead> Iterator for WiderAnnotations<R> {
    type Item = io::Result<AnnotatedImage>;

    fn next(&mut self) -> Option<Self::Item> {
        // Block header: the image path (blank lines are skipped).
        let image_file_path = loop {
            let line = self.read_line()?;
            match line {
                Err(e) => return Some(Err(e)),
                Ok(l) => {
                    let l = l.trim();
                    if !l.is_empty() {
                        break l.to_string();
                    }
                }
            }
        };
        if !image_file_path.contains('/') || !image_file_path.ends_with(".jpg") {
            return Some(Err(invalid_data(format!(
                "line {}: not an image path: {image_file_path:?}",
                self.line_no
            ))));
        }

        // Face count line.
        let count_line = match self.read_line() {
            None => {
                return Some(Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "line {}: missing face count after {image_file_path:?}",
                        self.line_no + 1
                    ),
                )));
            }
            Some(Err(e)) => return Some(Err(e)),
            Some(Ok(l)) => l,
        };
        let count: usize = match count_line.trim().parse() {
            Ok(c) => c,
            Err(_) => {
                return Some(Err(invalid_data(format!(
                    "line {}: invalid face count {count_line:?}",
                    self.line_no
                ))));
            }
        };

        // One 10-integer line per face.
        let mut faces = Vec::with_capacity(count);
        for _ in 0..count {
            let line = match self.read_line() {
                None => {
                    return Some(Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!(
                            "line {}: expected {count} face lines for {image_file_path:?}",
                            self.line_no + 1
                        ),
                    )));
                }
                Some(Err(e)) => return Some(Err(e)),
                Some(Ok(l)) => l,
            };
            match parse_face_box(self.line_no, &line) {
                Ok(b) => faces.push(b),
                Err(e) => return Some(Err(e)),
            }
        }

        // Zero-face images carry one all-zero line to consume.
        if count == 0 {
            if let Some(Ok(l)) = self.lines.peek() {
                if is_zero_line(l) {
                    self.lines.next();
                    self.line_no += 1;
                }
            }
        }

        Some(Ok(AnnotatedImage {
            image_file_path,
            faces,
        }))
    }
}

fn parse_face_box(line_no: usize, line: &str) -> io::Result<FaceBox> {
    let fields: Vec<u32> = line
        .split_whitespace()
        .map(|f| f.parse::<u32>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| {
            invalid_data(format!(
                "line {line_no}: invalid integer in box line {line:?}"
            ))
        })?;
    if fields.len() != 10 {
        return Err(invalid_data(format!(
            "line {line_no}: expected 10 fields, got {} in {line:?}",
            fields.len()
        )));
    }
    Ok(FaceBox {
        x: fields[0],
        y: fields[1],
        width: fields[2],
        height: fields[3],
        blur: fields[4],
        expression: fields[5],
        illumination: fields[6],
        invalid: fields[7],
        occlusion: fields[8],
        pose: fields[9],
    })
}

fn is_zero_line(line: &str) -> bool {
    let fields: Vec<&str> = line.split_whitespace().collect();
    fields.len() == 10 && fields.iter().all(|f| *f == "0")
}

fn invalid_data(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_preflight_identifies_missing_paths() -> io::Result<()> {
        let dir = tempfile::tempdir()?;
        let annotations = dir.path().join("annotations.txt");
        let images = dir.path().join("images");
        let error = load_training_samples(&annotations, &images).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert!(
            error
                .to_string()
                .contains(&annotations.display().to_string())
        );
        std::fs::write(&annotations, "event/sample.jpg\n1\n1 2 3 4 0 0 0 0 0 0\n")?;
        let error = load_training_samples(&annotations, &images).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert!(error.to_string().contains("IMAGES_ROOT"));
        assert!(error.to_string().contains(&images.display().to_string()));
        std::fs::create_dir_all(images.join("event"))?;
        let image = images.join("event/sample.jpg");
        let error = load_training_samples(&annotations, &images).unwrap_err();
        assert!(error.to_string().contains(&image.display().to_string()));
        // La prévalidation vérifie la présence ; le décodage reste fait par lot.
        std::fs::write(&image, b"fixture")?;
        let samples = load_training_samples(&annotations, &images)?;
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].image_file_path, "event/sample.jpg");
        Ok(())
    }

    #[test]
    fn training_preflight_propagates_malformed_annotations() -> io::Result<()> {
        let dir = tempfile::tempdir()?;
        let annotations = dir.path().join("annotations.txt");
        std::fs::write(&annotations, "event/sample.jpg\n1\ninvalid box\n")?;
        let error = load_training_samples(&annotations, dir.path()).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("invalid integer"));
        Ok(())
    }

    #[test]
    fn validation_loads_full_split_and_reports_its_own_paths() -> io::Result<()> {
        let dir = tempfile::tempdir()?;
        let annotations = dir.path().join("val.txt");
        let images = dir.path().join("WIDER_val/images");
        // Plus de 100 images et plus de 20 visages : aucune ancienne limite du prototype.
        let sample = format!(
            "event/sample.jpg\n21\n{}",
            "1 2 3 4 0 0 0 0 0 0\n".repeat(21)
        );
        std::fs::write(&annotations, sample.repeat(101))?;
        let error = load_validation_samples(&annotations, &images).unwrap_err();
        assert!(error.to_string().contains("WIDER_val.zip"));
        assert!(error.to_string().contains("VALID_IMAGES_ROOT"));
        std::fs::create_dir_all(images.join("event"))?;
        std::fs::write(images.join("event/sample.jpg"), b"fixture")?;
        let samples = load_validation_samples(&annotations, &images)?;
        assert_eq!(samples.len(), 101);
        assert!(samples.iter().all(|sample| sample.faces.len() == 21));
        assert_eq!(load_training_samples(&annotations, &images)?.len(), 101);
        Ok(())
    }
}
