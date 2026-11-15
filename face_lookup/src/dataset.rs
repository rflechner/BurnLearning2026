use std::fs::File;
use std::io::{self, BufRead, BufReader, Lines};
use std::iter::Peekable;
use std::path::{Path, PathBuf};

/// One face annotation line: `x y w h blur expression illumination invalid occlusion pose`.
/// All original attributes are kept so the filtering policy can change later.
#[derive(Debug, Clone)]
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

pub(crate) const ANNOTATIONS_PATH: &str =
    "data/extracted/db171f1b7fedf4d3453e81297ff02f9915356d19/wider_face_split/wider_face_train_bbx_gt.txt";

/// Annotation paths are relative to `WIDER_train/images/`, next to `wider_face_split/`.
pub fn images_root() -> PathBuf {
    Path::new(ANNOTATIONS_PATH)
        .parent()
        .and_then(|p| p.parent())
        .map(|extracted| extracted.join("WIDER_train/images"))
        .expect("annotations path has two parent directories")
}

/// Opens an annotation file and returns its lazy block iterator.
pub fn load_annotations(path: impl AsRef<Path>) -> io::Result<WiderAnnotations<BufReader<File>>> {
    let file = File::open(path)?;
    Ok(WiderAnnotations::new(BufReader::new(file)))
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
                )))
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
                ))))
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
                    )))
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
        .map_err(|_| invalid_data(format!("line {line_no}: invalid integer in box line {line:?}")))?;
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
