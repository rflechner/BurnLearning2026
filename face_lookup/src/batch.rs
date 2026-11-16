use burn::tensor::{Device, Tensor, TensorData};
use image::imageops::FilterType;
use std::path::{Component, Path};

use crate::detection_config::DetectionConfig;

/// Pas de sortie de l’architecture, indépendant de la résolution d’entrée.
pub const STRIDE: f32 = crate::detection_config::STRIDE as f32;

/// Prétraitement partagé par l'entraînement et l'inférence.
pub fn prepare_image(
    original: &image::RgbImage,
    config: DetectionConfig,
) -> (Vec<f32>, u32, u32, usize, usize) {
    let (ow, oh) = original.dimensions();
    let image_size = config.image_size();
    let plane = image_size * image_size;
    let mut pixels = vec![0.0f32; 3 * plane];
    let scale = (image_size as f32 / ow as f32).min(image_size as f32 / oh as f32);
    let width = ((ow as f32 * scale).round() as u32).clamp(1, image_size as u32);
    let height = ((oh as f32 * scale).round() as u32).clamp(1, image_size as u32);
    let left = (image_size - width as usize) / 2;
    let top = (image_size - height as usize) / 2;
    let resized = image::imageops::resize(original, width, height, FilterType::Triangle);
    for (x, y, pixel) in resized.enumerate_pixels() {
        for channel in 0..3 {
            pixels[channel * plane + (top + y as usize) * image_size + left + x as usize] =
                pixel[channel] as f32 / 255.0;
        }
    }
    (pixels, width, height, left, top)
}

pub struct DetectionBatch {
    pub images: Tensor<4>, // [N, 3, image_size, image_size], RGB en plans séparés
    /// Gaussienne centrée sur chaque cellule positive (1 au centre). Les voisines voient
    /// presque les mêmes pixels : elles restent négatives, mais pénalisées moins fort.
    pub heatmap: Tensor<4>, // [N, 1, grid_size, grid_size]
    /// dx, dy dans la cellule, puis ln(w / 8), ln(h / 8) : le log rend l'erreur relative,
    /// donc comparable entre un visage de 5 px et un de 200 px.
    pub boxes: Tensor<4>, // [N, 4, grid_size, grid_size]
    pub positive_mask: Tensor<4>, // [N, 1, grid_size, grid_size], 1 à la cellule du centre d'un visage retenu
    pub valid_mask: Tensor<4>,    // cellules utilisables pour la présence
    pub collisions: usize,
}

/// Prépare un lot à la fois, sans charger tout le dataset sur le GPU.
/// En cas de collision, garde le premier visage dans l'ordre des annotations.
pub fn make_batch(
    entries: &[crate::dataset::AnnotatedImage],
    images_root: &Path,
    device: &Device,
    config: DetectionConfig,
) -> Result<DetectionBatch, Box<dyn std::error::Error>> {
    if entries.is_empty() {
        return Err("Un lot ne peut pas être vide".into());
    }
    let image_size = config.image_size();
    let grid_size = config.grid_size();
    let cells = grid_size * grid_size;
    let n = entries.len();
    let plane = image_size * image_size;
    let mut pixels = vec![0.0f32; n * 3 * plane];
    let mut presence = vec![0.0f32; n * cells];
    let mut heatmap = vec![0.0f32; n * cells];
    let mut boxes = vec![0.0f32; n * 4 * cells];
    let mut valid = vec![1.0f32; n * cells];
    let mut collisions = 0;

    for (i, entry) in entries.iter().enumerate() {
        let relative = Path::new(&entry.image_file_path);
        if relative
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err(format!("Chemin image invalide : {}", relative.display()).into());
        }
        let path = images_root.join(relative);
        let original = image::open(&path)
            .map_err(|error| format!("Impossible de lire l'image {} : {error}", path.display()))?
            .to_rgb8();
        let (ow, oh) = original.dimensions();
        if ow == 0 || oh == 0 {
            return Err("Image vide".into());
        }
        let (image_pixels, width, height, left, top) = prepare_image(&original, config);
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
            for y in (y1 / STRIDE).floor() as usize..((y2 / STRIDE).ceil() as usize).min(grid_size)
            {
                for x in
                    (x1 / STRIDE).floor() as usize..((x2 / STRIDE).ceil() as usize).min(grid_size)
                {
                    valid[i * cells + y * grid_size + x] = 0.0;
                }
            }
        }
        for &(invalid, x1, y1, x2, y2) in &transformed {
            if invalid || x2 <= x1 || y2 <= y1 {
                continue;
            }
            let cx = (x1 + x2) / (2.0 * STRIDE);
            let cy = (y1 + y2) / (2.0 * STRIDE);
            let x = (cx.floor() as usize).min(grid_size - 1);
            let y = (cy.floor() as usize).min(grid_size - 1);
            let cell = y * grid_size + x;
            if presence[i * cells + cell] == 1.0 {
                collisions += 1;
                continue;
            }
            presence[i * cells + cell] = 1.0;
            valid[i * cells + cell] = 1.0;
            for (channel, value) in [
                cx - x as f32,
                cy - y as f32,
                ((x2 - x1) / STRIDE).ln(),
                ((y2 - y1) / STRIDE).ln(),
            ]
            .into_iter()
            .enumerate()
            {
                boxes[(i * 4 + channel) * cells + cell] = value;
            }
            // σ = taille / 6 : la gaussienne est quasi nulle au bord de la boîte.
            let sigma_x = (x2 - x1) / STRIDE / 6.0;
            let sigma_y = (y2 - y1) / STRIDE / 6.0;
            let range = |center: usize, sigma: f32| {
                let reach = (3.0 * sigma).ceil() as usize;
                center.saturating_sub(reach)..(center + reach + 1).min(grid_size)
            };
            for gy in range(y, sigma_y) {
                for gx in range(x, sigma_x) {
                    let (dx, dy) = (gx as f32 - x as f32, gy as f32 - y as f32);
                    let value = (-(dx * dx) / (2.0 * sigma_x * sigma_x)
                        - (dy * dy) / (2.0 * sigma_y * sigma_y))
                        .exp();
                    let target = &mut heatmap[i * cells + gy * grid_size + gx];
                    *target = target.max(value);
                }
            }
        }
    }
    let mask_shape = [n, 1, grid_size, grid_size];
    let presence = Tensor::from_data(TensorData::new(presence, mask_shape), device);
    Ok(DetectionBatch {
        images: Tensor::from_data(
            TensorData::new(pixels, [n, 3, image_size, image_size]),
            device,
        ),
        positive_mask: presence,
        heatmap: Tensor::from_data(TensorData::new(heatmap, mask_shape), device),
        boxes: Tensor::from_data(TensorData::new(boxes, [n, 4, grid_size, grid_size]), device),
        valid_mask: Tensor::from_data(TensorData::new(valid, mask_shape), device),
        collisions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    const IMAGE_SIZE: usize = 320;
    const GRID_SIZE: usize = 40;
    const CELLS: usize = GRID_SIZE * GRID_SIZE;
    use crate::{
        dataset::{AnnotatedImage, FaceBox},
        model::Model,
    };
    use burn::{
        module::{Module, ModuleVisitor, Param},
        optim::{AdamConfig, GradientsParams},
        store::ModuleRecord,
    };

    #[derive(Default)]
    struct Weights(Vec<Vec<f32>>);

    impl ModuleVisitor for Weights {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            self.0
                .push(param.val().to_data().try_to_vec::<f32>().unwrap());
        }
    }

    #[test]
    fn high_resolution_training_checkpoint_and_inference() -> Result<(), Box<dyn std::error::Error>>
    {
        let dir = tempfile::tempdir()?;
        let image = image::RgbImage::from_pixel(640, 320, image::Rgb([100, 150, 200]));
        image.save(dir.path().join("sample.png"))?;
        let face = FaceBox {
            x: 190,
            y: 200,
            width: 12,
            height: 16,
            blur: 0,
            expression: 0,
            illumination: 0,
            invalid: 0,
            occlusion: 0,
            pose: 0,
        };
        let mut neighbor = face.clone();
        neighbor.x += 8;
        let entries = [AnnotatedImage {
            image_file_path: "sample.png".into(),
            faces: vec![face, neighbor],
        }];
        let device = crate::device::default_device().autodiff();
        let low = make_batch(&entries, dir.path(), &device, DetectionConfig::default())?;
        assert_eq!(low.collisions, 1);
        let config = DetectionConfig::new(640)?;
        let batch = make_batch(&entries, dir.path(), &device, config)?;
        assert_eq!(batch.images.dims(), [1, 3, 640, 640]);
        assert_eq!(batch.boxes.dims(), [1, 4, 80, 80]);
        assert_eq!(batch.collisions, 0);
        assert_eq!(batch.positive_mask.clone().sum().into_scalar::<f32>(), 2.0);
        // Les poids 320 ont exactement les mêmes dimensions : transfert possible en 640.
        let legacy = dir.path().join("legacy.bpk");
        Model::default(&device).into_record().save(&legacy)?;
        let model = Model::default(&device).try_load_record(ModuleRecord::load(&legacy)?)?;
        let before = model.forward(batch.images.clone());
        assert_eq!(before.dims(), [1, 5, 80, 80]);
        let loss = crate::loss::detection_loss(before.clone(), &batch);
        let diagnostics = loss.diagnostics(&batch)?;
        assert!(diagnostics.total.is_finite());
        assert!(diagnostics.to_string().contains("entrée 640×640"));
        assert!(diagnostics.to_string().contains("attendue=12.00×16.00px"));
        let grads = GradientsParams::from_grads(loss.total.backward(), &model);
        let model = AdamConfig::new().init().step(1e-3, model, grads);
        let after = model.forward(batch.images.clone());
        let change = (after.clone() - before).abs().sum().into_scalar::<f32>();
        assert!(change.is_finite() && change > 0.0);
        let mut expected_weights = Weights::default();
        model.visit(&mut expected_weights);
        let path = dir.path().join("model.bpk");
        config.save_for_weights(&path)?;
        model.into_record().save(&path)?;
        let loaded = crate::infer::load(&path)?;
        assert_eq!(loaded.config, config);
        // Vérifier la persistance bit à bit. Les kernels choisis pour l'autodiff et
        // l'inférence peuvent avoir des arrondis différents sur leurs sorties.
        let mut restored_weights = Weights::default();
        loaded.network.visit(&mut restored_weights);
        assert!(!expected_weights.0.is_empty());
        assert_eq!(restored_weights.0, expected_weights.0);
        let inference_batch = make_batch(&entries, dir.path(), &device.inner(), config)?;
        let restored = loaded.network.forward(inference_batch.images);
        assert_eq!(restored.dims(), [1, 5, 80, 80]);
        assert!(
            restored
                .into_data()
                .try_to_vec::<f32>()?
                .iter()
                .all(|v| v.is_finite())
        );
        let detections = crate::infer::predict_image(&loaded, &image, 0.5)?;
        assert!(
            detections
                .iter()
                .all(|d| d.x1 >= 0.0 && d.y1 >= 0.0 && d.x2 <= 640.0 && d.y2 <= 320.0)
        );
        Ok(())
    }

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
        let batch = make_batch(&samples, dir.path(), &device, DetectionConfig::default())?;
        assert_eq!(batch.images.dims(), [1, 3, 320, 320]);
        assert_eq!(batch.collisions, 1);
        assert_eq!(batch.positive_mask.clone().sum().into_scalar::<f32>(), 1.0);
        // Letterbox : image 320x160, marge verticale 80, centre (112,144).
        let cell = 18 * GRID_SIZE + 14;
        let targets = batch.boxes.to_data().try_to_vec::<f32>()?;
        assert_eq!(targets[cell], 0.0);
        assert_eq!(targets[CELLS + cell], 0.0);
        // Boîte 64×48 px dans l'entrée : ln(64 / 8), ln(48 / 8).
        assert!((targets[2 * CELLS + cell] - 8f32.ln()).abs() < 1e-6);
        assert!((targets[3 * CELLS + cell] - 6f32.ln()).abs() < 1e-6);
        let heat = batch.heatmap.to_data().try_to_vec::<f32>()?;
        assert_eq!(heat[cell], 1.0);
        // Voisine de droite : σx = 8 cellules / 6, donc exp(-1 / (2σx²)).
        let sigma_x = 8.0f32 / 6.0;
        assert!((heat[cell + 1] - (-1.0 / (2.0 * sigma_x * sigma_x)).exp()).abs() < 1e-6);
        assert_eq!(heat[0], 0.0);
        let pixels = batch.images.to_data().try_to_vec::<f32>()?;
        assert_eq!(pixels[0], 0.0);
        assert_eq!(pixels[80 * IMAGE_SIZE], 1.0);
        assert!((pixels[IMAGE_SIZE * IMAGE_SIZE + 80 * IMAGE_SIZE] - 128.0 / 255.0).abs() < 1e-6);
        let mut model = Model::default(&device);
        let before = model.forward(batch.images.clone());
        assert_eq!(before.dims(), [1, 5, 40, 40]);
        let loss = crate::loss::detection_loss(before.clone(), &batch);
        assert!(loss.total.clone().into_scalar::<f32>().is_finite());
        let diagnostics = loss.diagnostics(&batch)?.to_string();
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
        let mut expected_weights = Weights::default();
        model.visit(&mut expected_weights);
        model.into_record().save(&path)?;
        let restored = Model::default(&device).load_record(ModuleRecord::load(&path)?);
        let mut restored_weights = Weights::default();
        restored.visit(&mut restored_weights);
        assert!(!expected_weights.0.is_empty());
        assert_eq!(restored_weights.0, expected_weights.0);
        let inference_model = crate::infer::load(&path).unwrap();
        let image_bytes = std::fs::read(dir.path().join("sample.png"))?;
        let detections = crate::infer::predict(&inference_model, &image_bytes, 0.5).unwrap();
        let overlay = crate::infer::overlay(&image_bytes, &samples[0].faces, &detections).unwrap();
        assert_eq!(image::load_from_memory(&overlay)?.width(), 160);
        assert!(
            restored
                .forward(batch.images.clone())
                .into_data()
                .try_to_vec::<f32>()?
                .iter()
                .all(|v| v.is_finite())
        );
        let empty = vec![AnnotatedImage {
            image_file_path: "sample.png".into(),
            faces: vec![],
        }];
        let batch = make_batch(&empty, dir.path(), &device, DetectionConfig::default())?;
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
                .to_string()
                .contains("tailles=n/a (aucun positif)")
        );
        Ok(())
    }
}
