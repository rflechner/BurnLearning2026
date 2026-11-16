//! Inférence : même letterboxing que l'entraînement, décodage puis NMS.
use crate::{detection_config::DetectionConfig, model::Model};
use burn::{
    module::Module,
    store::ModuleRecord,
    tensor::{Tensor, TensorData},
};
use std::path::Path;

pub struct InferenceModel {
    pub network: Model,
    pub config: DetectionConfig,
}

#[derive(Clone, Debug)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

pub fn predict(
    model: &InferenceModel,
    bytes: &[u8],
    threshold: f32,
) -> Result<Vec<Detection>, String> {
    let image = image::load_from_memory(bytes)
        .map_err(|e| e.to_string())?
        .to_rgb8();
    predict_image(model, &image, threshold)
}

/// Plus grand côté conservé pour une photo envoyée, supérieur aux résolutions du réseau.
/// Dessiner puis encoder en PNG une photo de 24 Mpx prendrait plusieurs secondes.
pub const UPLOAD_MAX_SIDE: u32 = 1600;

/// Décode une photo envoyée par l'utilisateur : applique l'orientation EXIF (sinon une
/// photo de téléphone en portrait arrive couchée) puis réduit sa taille d'affichage.
pub fn load_upload(bytes: &[u8]) -> Result<image::RgbImage, String> {
    use image::ImageDecoder;
    let mut decoder = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| e.to_string())?
        .into_decoder()
        .map_err(|e| e.to_string())?;
    let orientation = decoder.orientation().map_err(|e| e.to_string())?;
    let mut image = image::DynamicImage::from_decoder(decoder).map_err(|e| e.to_string())?;
    image.apply_orientation(orientation);
    if image.width().max(image.height()) > UPLOAD_MAX_SIDE {
        image = image.resize(
            UPLOAD_MAX_SIDE,
            UPLOAD_MAX_SIDE,
            image::imageops::FilterType::Triangle,
        );
    }
    Ok(image.to_rgb8())
}

pub fn predict_image(
    model: &InferenceModel,
    image: &image::RgbImage,
    threshold: f32,
) -> Result<Vec<Detection>, String> {
    let (ow, oh) = image.dimensions();
    let device = crate::device::default_device();
    let config = model.config;
    let size = config.image_size();
    let (pixels, width, height, left, top) = crate::batch::prepare_image(image, config);
    let input = Tensor::from_data(TensorData::new(pixels, [1, 3, size, size]), &device);
    let output = model
        .network
        .forward(input)
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|e| e.to_string())?;
    decode(&output, ow, oh, width, height, left, top, threshold, config)
}

pub fn load(path: &Path) -> Result<InferenceModel, String> {
    let config = DetectionConfig::for_weights(path).map_err(|e| e.to_string())?;
    let record = ModuleRecord::load(path).map_err(|e| e.to_string())?;
    let network = Model::default(&crate::device::default_device())
        .try_load_record(record)
        .map_err(|e| e.to_string())?;
    Ok(InferenceModel { network, config })
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
fn iou(a: &Detection, b: &Detection) -> f32 {
    let intersection =
        (a.x2.min(b.x2) - a.x1.max(b.x1)).max(0.0) * (a.y2.min(b.y2) - a.y1.max(b.y1)).max(0.0);
    intersection
        / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection).max(1e-8)
}
#[allow(clippy::too_many_arguments)]
fn decode(
    values: &[f32],
    ow: u32,
    oh: u32,
    width: u32,
    height: u32,
    left: usize,
    top: usize,
    threshold: f32,
    config: DetectionConfig,
) -> Result<Vec<Detection>, String> {
    let grid = config.grid_size();
    let cells = grid * grid;
    if values.len() != 5 * cells || values.iter().any(|v| !v.is_finite()) {
        return Err("Sortie du modèle invalide".into());
    }
    let mut candidates = Vec::new();
    for cell in 0..cells {
        let score = sigmoid(values[cell]);
        if score < threshold {
            continue;
        }
        let cx = ((cell % grid) as f32 + sigmoid(values[cells + cell])) * crate::batch::STRIDE;
        let cy = ((cell / grid) as f32 + sigmoid(values[2 * cells + cell])) * crate::batch::STRIDE;
        // Tailles prédites en ln(px / 8) ; borner avant exp pour éviter l'infini.
        let stride = crate::batch::STRIDE;
        let w = values[3 * cells + cell].min(8.0).exp() * stride;
        let h = values[4 * cells + cell].min(8.0).exp() * stride;
        let sx = width as f32 / ow as f32;
        let sy = height as f32 / oh as f32;
        let d = Detection {
            x1: ((cx - w / 2.0 - left as f32) / sx).clamp(0.0, ow as f32),
            y1: ((cy - h / 2.0 - top as f32) / sy).clamp(0.0, oh as f32),
            x2: ((cx + w / 2.0 - left as f32) / sx).clamp(0.0, ow as f32),
            y2: ((cy + h / 2.0 - top as f32) / sy).clamp(0.0, oh as f32),
            score,
        };
        if d.x2 > d.x1 && d.y2 > d.y1 {
            candidates.push(d);
        }
    }
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::new();
    for d in candidates {
        if kept.iter().all(|other| iou(&d, other) <= 0.4) {
            kept.push(d);
        }
    }
    Ok(kept)
}

pub fn overlay(
    bytes: &[u8],
    faces: &[crate::dataset::FaceBox],
    detections: &[Detection],
) -> Result<Vec<u8>, String> {
    let image = image::load_from_memory(bytes)
        .map_err(|e| e.to_string())?
        .to_rgb8();
    overlay_image(image, faces, detections)
}

/// Dessine annotations (vert) et prédictions (rouge), puis encode en PNG.
pub fn overlay_image(
    mut image: image::RgbImage,
    faces: &[crate::dataset::FaceBox],
    detections: &[Detection],
) -> Result<Vec<u8>, String> {
    crate::render::draw_face_boxes(&mut image, faces);
    let (w, h) = image.dimensions();
    if w == 0 || h == 0 {
        return Err("Image vide".into());
    }
    for d in detections {
        let (x1, y1, x2, y2) = (
            d.x1.floor() as u32,
            d.y1.floor() as u32,
            (d.x2.ceil() as u32).saturating_sub(1).min(w - 1),
            (d.y2.ceil() as u32).saturating_sub(1).min(h - 1),
        );
        for t in 0..2 {
            if x1 + t > x2 || y1 + t > y2 {
                break;
            }
            for x in x1..=x2 {
                image.put_pixel(x, y1 + t, image::Rgb([255, 65, 65]));
                image.put_pixel(x, y2.saturating_sub(t), image::Rgb([255, 65, 65]));
            }
            for y in y1..=y2 {
                image.put_pixel(x1 + t, y, image::Rgb([255, 65, 65]));
                image.put_pixel(x2.saturating_sub(t), y, image::Rgb([255, 65, 65]));
            }
        }
    }
    let mut out = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut out, image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(out.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decodes_high_resolution_grid_and_letterbox() {
        let config = DetectionConfig::new(640).unwrap();
        let cells = 80 * 80;
        let mut values = vec![-100.0; 5 * cells];
        let cell = 40 * 80 + 60;
        values[cell] = 10.0;
        values[cells + cell] = 0.0;
        values[2 * cells + cell] = 0.0;
        values[3 * cells + cell] = 8f32.ln();
        values[4 * cells + cell] = 6f32.ln();
        let detections = decode(&values, 320, 160, 640, 320, 0, 160, 0.5, config).unwrap();
        assert_eq!(detections.len(), 1);
        let d = &detections[0];
        for (actual, expected) in [(d.x1, 226.0), (d.y1, 70.0), (d.x2, 258.0), (d.y2, 94.0)] {
            assert!((actual - expected).abs() < 1e-4);
        }
        assert!(
            decode(
                &values,
                320,
                160,
                640,
                320,
                0,
                160,
                0.5,
                DetectionConfig::default()
            )
            .is_err()
        );
    }

    #[test]
    fn decodes_letterbox_and_filters_background() {
        let mut values = vec![-100.0; 8000];
        let cell = 18 * 40 + 14;
        values[cell] = 10.0;
        values[1600 + cell] = 0.0;
        values[3200 + cell] = 0.0;
        values[4800 + cell] = 8f32.ln();
        values[6400 + cell] = 6f32.ln();
        let detections = decode(
            &values,
            160,
            80,
            320,
            160,
            0,
            80,
            0.5,
            DetectionConfig::default(),
        )
        .unwrap();
        assert_eq!(detections.len(), 1);
        let d = &detections[0];
        assert!((d.x1 - 42.0).abs() < 0.001);
        assert!((d.y1 - 22.0).abs() < 0.001);
        assert!((d.x2 - 74.0).abs() < 0.001);
        assert!((d.y2 - 46.0).abs() < 0.001);
        assert!(
            decode(
                &values,
                160,
                80,
                320,
                160,
                0,
                80,
                1.0,
                DetectionConfig::default()
            )
            .unwrap()
            .is_empty()
        );
        values[0] = f32::NAN;
        assert!(
            decode(
                &values,
                160,
                80,
                320,
                160,
                0,
                80,
                0.5,
                DetectionConfig::default()
            )
            .is_err()
        );
    }
    #[test]
    fn nms_removes_overlapping_predictions() {
        let mut values = vec![-100.0; 8000];
        for cell in [18 * 40 + 14, 18 * 40 + 15] {
            values[cell] = 10.0;
            values[1600 + cell] = 0.0;
            values[3200 + cell] = 0.0;
            values[4800 + cell] = 8f32.ln();
            values[6400 + cell] = 6f32.ln();
        }
        assert_eq!(
            decode(
                &values,
                160,
                80,
                320,
                160,
                0,
                80,
                0.5,
                DetectionConfig::default()
            )
            .unwrap()
            .len(),
            1
        );
    }
    #[test]
    fn overlay_draws_red_predictions() {
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::RgbImage::new(20, 20)
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        let result = overlay(
            bytes.get_ref(),
            &[],
            &[Detection {
                x1: 2.0,
                y1: 3.0,
                x2: 10.0,
                y2: 12.0,
                score: 0.9,
            }],
        )
        .unwrap();
        let image = image::load_from_memory(&result).unwrap().to_rgb8();
        assert_eq!(image.get_pixel(2, 3).0, [255, 65, 65]);
        assert_eq!(image.get_pixel(5, 6).0, [0, 0, 0]);
    }
    #[test]
    fn upload_applies_exif_orientation_and_limits_size() {
        let mut jpeg = std::io::Cursor::new(Vec::new());
        image::RgbImage::new(40, 20)
            .write_to(&mut jpeg, image::ImageFormat::Jpeg)
            .unwrap();
        let jpeg = jpeg.into_inner();
        // Segment APP1 minimal : TIFF big-endian, une entrée Orientation = 6 (rotation 90° horaire).
        let mut app1 = vec![0xFF, 0xE1, 0x00, 0x22];
        app1.extend_from_slice(b"Exif\0\0MM\0\x2a\0\0\0\x08");
        app1.extend_from_slice(&[0, 1, 0x01, 0x12, 0, 3, 0, 0, 0, 1, 0, 6, 0, 0, 0, 0, 0, 0]);
        let rotated = [&jpeg[..2], &app1, &jpeg[2..]].concat();
        let image = load_upload(&rotated).unwrap();
        assert_eq!(image.dimensions(), (20, 40));

        let mut png = std::io::Cursor::new(Vec::new());
        image::RgbImage::new(3200, 1000)
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let image = load_upload(png.get_ref()).unwrap();
        assert_eq!(image.dimensions(), (UPLOAD_MAX_SIDE, 500));

        assert!(load_upload(b"pas une image").is_err());
    }
}
