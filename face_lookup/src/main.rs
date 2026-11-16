mod api;
mod batch;
mod dataset;
mod device;
mod infer;
mod loss;
mod model;
mod render;

use crate::batch::make_batch;
use crate::loss::detection_loss;
use crate::model::Model;
use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams};
use device::default_device;

/// How many images to fully decode; the rest are only checked for existence.
const DECODE_LIMIT: usize = 50;
/// Dossier des poids entraînés, sous `runs/` pour apparaître dans le viewer.
const RUN_DIR: &str = "runs/dilated";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let annotations_path = std::env::var_os("ANNOTATIONS_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(dataset::ANNOTATIONS_PATH)
        });
    let images_root_path = std::env::var_os("IMAGES_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(dataset::images_root())
        });

    // Train from dataset
    if std::env::args().any(|arg| arg == "--train") {
        train_dataset(&annotations_path, &images_root_path)?;
        return Ok(());
    }

    // Inspect dataset as a simple console app
    if std::env::args().any(|arg| arg == "--inspect") {
        inspect_dataset(&*annotations_path, &*images_root_path);
        return Ok(());
    }

    // Web server
    let state = api::DatasetState::load(&annotations_path, &images_root_path)?;
    let address = std::env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:3001".into());
    let listener = tokio::net::TcpListener::bind(&address).await?;
    println!("Dataset API: http://{}", listener.local_addr()?);
    axum::serve(listener, api::router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

fn train_dataset(
    annotations_path: &std::path::Path,
    images_root: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = default_device().autodiff();
    println!("Using device: {:?}", device);
    device.seed(42);

    let mut model = Model::default(&device);
    let mut optimizer = AdamConfig::new().init();

    // Petit ensemble pour commencer.
    let samples = dataset::load_annotations(annotations_path)?
        .filter(|sample| {
            if let Ok(item) = sample {
                item.faces.len() <= 20
            } else {
                false
            }
        })
        .take(100)
        .collect::<Result<Vec<_>, _>>()?;

    if samples.is_empty() {
        return Err("Aucune image d'entraînement".into());
    }
    // Nouveau dossier : les poids de runs/first ont l'ancienne architecture et ne se chargent plus.
    std::fs::create_dir_all(RUN_DIR)?;

    let batches = samples.len().div_ceil(4);
    let total_steps = EPOCHS * batches;
    for epoch in 0..EPOCHS {
        for (batch_index, entries) in samples.chunks(4).enumerate() {
            let batch = make_batch(entries, images_root, &device)?;

            let predictions = model.forward(batch.images.clone());
            let loss = detection_loss(predictions, &batch);
            let lr = learning_rate(epoch * batches + batch_index, total_steps);

            println!(
                "Époque {} · lot {}/{} : {}, collisions={}, lr={lr:.2e}",
                epoch + 1,
                batch_index + 1,
                batches,
                loss.diagnostics(&batch)?,
                batch.collisions
            );
            let gradients = loss.total.backward();
            let gradients = GradientsParams::from_grads(gradients, &model);
            model = optimizer.step(lr, model, gradients);
        }
        model.clone().into_record().save(format!("{RUN_DIR}/model.bpk"))?;
    }

    Ok(())
}

const EPOCHS: usize = 100;
const LR_MAX: f64 = 1e-3;
const LR_MIN: f64 = 1e-5;

/// Décroissance cosinus de `LR_MAX` à `LR_MIN` sur tout l'entraînement.
/// À taux constant, Adam et la L1 (gradient de norme fixe) font osciller l'échelle des
/// tailles d'un lot à l'autre (±40 % mesurés) ; réduire le pas la laisse se stabiliser.
fn learning_rate(step: usize, total_steps: usize) -> f64 {
    let progress = step as f64 / total_steps.max(1) as f64;
    LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + (std::f64::consts::PI * progress).cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learning_rate_decays_from_max_to_min() {
        assert_eq!(learning_rate(0, 100), LR_MAX);
        assert!((learning_rate(50, 100) - (LR_MAX + LR_MIN) / 2.0).abs() < 1e-12);
        assert!((learning_rate(100, 100) - LR_MIN).abs() < 1e-12);
        assert!(learning_rate(30, 100) > learning_rate(31, 100));
    }
}

fn inspect_dataset(annotations_path: &std::path::Path, images_root: &std::path::Path) {
    let train_data = dataset::load_annotations(annotations_path).expect("open annotations");
    let mut checked = 0;
    let mut faces = 0;
    let mut missing = 0;
    let mut decoded = 0;
    let mut errors = 0;
    for entry in train_data {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                errors += 1;
                eprintln!("annotation error: {e}");
                continue;
            }
        };
        checked += 1;
        faces += entry.faces.len();

        let path = images_root.join(&entry.image_file_path);
        if !path.exists() {
            missing += 1;
            eprintln!("missing image: {}", entry.image_file_path);
            continue;
        }
        if decoded < DECODE_LIMIT {
            match image::open(&path) {
                Ok(img) => {
                    println!(
                        "decoded {} -> {}x{} px, {} face(s)",
                        entry.image_file_path,
                        img.width(),
                        img.height(),
                        entry.faces.len()
                    );
                    decoded += 1;
                }
                Err(e) => eprintln!("failed to decode {}: {e}", entry.image_file_path),
            }
        }
    }
    println!(
        "checked={checked} faces={faces} decoded={decoded} missing={missing} annotation_errors={errors}"
    );
}
