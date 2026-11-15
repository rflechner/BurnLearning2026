use burn::tensor::{Device, Tensor, TensorData};
use image::imageops::FilterType;
use std::path::{Component, Path};

pub const IMAGE_SIZE: usize = 320;
pub const GRID_SIZE: usize = 40;
const CELLS: usize = GRID_SIZE * GRID_SIZE;

/// Prétraitement partagé par l'entraînement et l'inférence.
pub fn prepare_image(original: &image::RgbImage) -> (Vec<f32>, u32, u32, usize, usize) {
    let (ow, oh) = original.dimensions();
    let plane = IMAGE_SIZE * IMAGE_SIZE;
    let mut pixels = vec![0.0f32; 3 * plane];
    let scale = (IMAGE_SIZE as f32 / ow as f32).min(IMAGE_SIZE as f32 / oh as f32);
    let width = ((ow as f32 * scale).round() as u32).clamp(1, IMAGE_SIZE as u32);
    let height = ((oh as f32 * scale).round() as u32).clamp(1, IMAGE_SIZE as u32);
    let left = (IMAGE_SIZE - width as usize) / 2;
    let top = (IMAGE_SIZE - height as usize) / 2;
    let resized = image::imageops::resize(original, width, height, FilterType::Triangle);
    for (x, y, pixel) in resized.enumerate_pixels() {
        for channel in 0..3 {
            pixels[channel * plane + (top + y as usize) * IMAGE_SIZE + left + x as usize] =
                pixel[channel] as f32 / 255.0;
        }
    }
    (pixels, width, height, left, top)
}

pub struct DetectionBatch {
    pub images: Tensor<4>,        // [N, 3, 320, 320], RGB en plans séparés
    pub presence: Tensor<4>,      // [N, 1, 40, 40]
    pub boxes: Tensor<4>,         // [N, 4, 40, 40], dx, dy, w/320, h/320
    pub positive_mask: Tensor<4>, // cellules avec un visage retenu
    pub valid_mask: Tensor<4>,    // cellules utilisables pour la présence
    pub collisions: usize,
}

/// Prépare un lot à la fois, sans charger tout le dataset sur le GPU.
/// En cas de collision, garde le premier visage dans l'ordre des annotations.
pub fn make_batch(
    entries: &[crate::dataset::AnnotatedImage],
    images_root: &Path,
    device: &Device,
) -> Result<DetectionBatch, Box<dyn std::error::Error>> {
    if entries.is_empty() {
        return Err("Un lot ne peut pas être vide".into());
    }
    let n = entries.len();
    let plane = IMAGE_SIZE * IMAGE_SIZE;
    let mut pixels = vec![0.0f32; n * 3 * plane];
    let mut presence = vec![0.0f32; n * CELLS];
    let mut boxes = vec![0.0f32; n * 4 * CELLS];
    let mut valid = vec![1.0f32; n * CELLS];
    let mut collisions = 0;

    for (i, entry) in entries.iter().enumerate() {
        let relative = Path::new(&entry.image_file_path);
        if relative
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err(format!("Chemin image invalide : {}", relative.display()).into());
        }
        let original = image::open(images_root.join(relative))?.to_rgb8();
        let (ow, oh) = original.dimensions();
        if ow == 0 || oh == 0 {
            return Err("Image vide".into());
        }
        let (image_pixels, width, height, left, top) = prepare_image(&original);
        pixels[i * 3 * plane..(i + 1) * 3 * plane].copy_from_slice(&image_pixels);
        // Utiliser les dimensions réellement arrondies, comme pour les pixels.
        let sx = width as f32 / ow as f32;
        let sy = height as f32 / oh as f32;
        let transformed: Vec<_> = entry
            .faces
            .iter()
            .map(|face| {
                let x1 = (face.x as f32).min(ow as f32) * sx + left as f32;
                let y1 = (face.y as f32).min(oh as f32) * sy + top as f32;
                let x2 = (face.x as f32 + face.width as f32).min(ow as f32) * sx + left as f32;
                let y2 = (face.y as f32 + face.height as f32).min(oh as f32) * sy + top as f32;
                (face.invalid != 0, x1, y1, x2, y2)
            })
            .collect();
        // Les cellules recouvrant une annotation invalide ne sont pas des négatifs.
        for &(invalid, x1, y1, x2, y2) in &transformed {
            if !invalid || x2 <= x1 || y2 <= y1 {
                continue;
            }
            for y in (y1 / 8.0).floor() as usize..((y2 / 8.0).ceil() as usize).min(GRID_SIZE) {
                for x in (x1 / 8.0).floor() as usize..((x2 / 8.0).ceil() as usize).min(GRID_SIZE) {
                    valid[i * CELLS + y * GRID_SIZE + x] = 0.0;
                }
            }
        }
        for &(invalid, x1, y1, x2, y2) in &transformed {
            if invalid || x2 <= x1 || y2 <= y1 {
                continue;
            }
            let cx = (x1 + x2) / 16.0;
            let cy = (y1 + y2) / 16.0;
            let x = (cx.floor() as usize).min(GRID_SIZE - 1);
            let y = (cy.floor() as usize).min(GRID_SIZE - 1);
            let cell = y * GRID_SIZE + x;
            if presence[i * CELLS + cell] == 1.0 {
                collisions += 1;
                continue;
            }
            presence[i * CELLS + cell] = 1.0;
            valid[i * CELLS + cell] = 1.0;
            for (channel, value) in [
                cx - x as f32,
                cy - y as f32,
                (x2 - x1) / IMAGE_SIZE as f32,
                (y2 - y1) / IMAGE_SIZE as f32,
            ]
            .into_iter()
            .enumerate()
            {
                boxes[(i * 4 + channel) * CELLS + cell] = value;
            }
        }
    }
    let mask_shape = [n, 1, GRID_SIZE, GRID_SIZE];
    let presence = Tensor::from_data(TensorData::new(presence, mask_shape), device);
    Ok(DetectionBatch {
        images: Tensor::from_data(
            TensorData::new(pixels, [n, 3, IMAGE_SIZE, IMAGE_SIZE]),
            device,
        ),
        positive_mask: presence.clone(),
        presence,
        boxes: Tensor::from_data(TensorData::new(boxes, [n, 4, GRID_SIZE, GRID_SIZE]), device),
        valid_mask: Tensor::from_data(TensorData::new(valid, mask_shape), device),
        collisions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dataset::{AnnotatedImage, FaceBox},
        model::Model,
    };
    use burn::{
        module::Module,
        optim::{AdamConfig, GradientsParams},
        store::ModuleRecord,
    };

    #[test]
    fn synthetic_batch_training_and_weight_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        image::RgbImage::from_pixel(160, 80, image::Rgb([255, 128, 0]))
            .save(dir.path().join("sample.png"))?;
        let face = FaceBox {
            x: 40,
            y: 20,
            width: 32,
            height: 24,
            blur: 0,
            expression: 0,
            illumination: 0,
            invalid: 0,
            occlusion: 0,
            pose: 0,
        };
        let samples = vec![AnnotatedImage {
            image_file_path: "sample.png".into(),
            faces: vec![face.clone(), face],
        }];
        let device = crate::device::default_device().autodiff();
        device.seed(42);
        let batch = make_batch(&samples, dir.path(), &device)?;
        assert_eq!(batch.images.dims(), [1, 3, 320, 320]);
        assert_eq!(batch.collisions, 1);
        assert_eq!(batch.presence.clone().sum().into_scalar::<f32>(), 1.0);
        // Letterbox : image 320x160, marge verticale 80, centre (112,144).
        let cell = 18 * GRID_SIZE + 14;
        let targets = batch.boxes.to_data().try_to_vec::<f32>()?;
        assert_eq!(targets[cell], 0.0);
        assert_eq!(targets[CELLS + cell], 0.0);
        assert!((targets[2 * CELLS + cell] - 0.2).abs() < 1e-6);
        assert!((targets[3 * CELLS + cell] - 0.15).abs() < 1e-6);
        let pixels = batch.images.to_data().try_to_vec::<f32>()?;
        assert_eq!(pixels[0], 0.0);
        assert_eq!(pixels[80 * IMAGE_SIZE], 1.0);
        assert!((pixels[IMAGE_SIZE * IMAGE_SIZE + 80 * IMAGE_SIZE] - 128.0 / 255.0).abs() < 1e-6);
        let mut model = Model::default(&device);
        let before = model.forward(batch.images.clone());
        assert_eq!(before.dims(), [1, 5, 40, 40]);
        let loss = crate::loss::detection_loss(before.clone(), &batch);
        assert!(loss.total.clone().into_scalar::<f32>().is_finite());
        let diagnostics = loss.diagnostics(&batch)?;
        assert!(diagnostics.contains("positifs=1"));
        assert!(diagnostics.contains("attendue=64.00×48.00px"));
        let total = loss.total.clone().into_scalar::<f32>();
        let parts = loss.presence.clone().into_scalar::<f32>()
            + loss.localization.clone().into_scalar::<f32>() * crate::loss::LOCALIZATION_WEIGHT;
        assert!((total - parts).abs() < 1e-5);
        let gradients = GradientsParams::from_grads(loss.total.backward(), &model);
        model = AdamConfig::new().init().step(1e-3, model, gradients);
        let after = model.forward(batch.images.clone());
        let change = (after.clone() - before).abs().sum().into_scalar::<f32>();
        assert!(change.is_finite() && change > 0.0);
        let path = dir.path().join("model.bpk");
        model.into_record().save(&path)?;
        let restored = Model::default(&device).load_record(ModuleRecord::load(&path)?);
        let inference_model = crate::infer::load(&path).unwrap();
        let image_bytes = std::fs::read(dir.path().join("sample.png"))?;
        let detections = crate::infer::predict(&inference_model, &image_bytes, 0.5).unwrap();
        let overlay = crate::infer::overlay(&image_bytes, &samples[0].faces, &detections).unwrap();
        assert_eq!(image::load_from_memory(&overlay)?.width(), 160);
        let difference = (restored.forward(batch.images.clone()) - after)
            .abs()
            .max()
            .into_scalar::<f32>();
        assert!(difference < 1e-5);
        let empty = vec![AnnotatedImage {
            image_file_path: "sample.png".into(),
            faces: vec![],
        }];
        let batch = make_batch(&empty, dir.path(), &device)?;
        assert!(
            crate::loss::detection_loss(restored.forward(batch.images.clone()), &batch)
                .total
                .into_scalar::<f32>()
                .is_finite()
        );
        let empty_loss =
            crate::loss::detection_loss(restored.forward(batch.images.clone()), &batch);
        assert!(
            empty_loss
                .diagnostics(&batch)?
                .contains("tailles=n/a (aucun positif)")
        );
        Ok(())
    }
}
