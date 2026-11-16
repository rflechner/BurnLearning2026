mod api;
mod batch;
mod dataset;
mod detection_config;
mod device;
mod infer;
mod loss;
mod model;
mod render;
mod training_failure;
mod training_restart;
mod training_ui;

use crate::batch::make_batch;
use crate::loss::detection_loss;
use crate::model::Model;
use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams};
use device::default_device;

/// How many images to fully decode; the rest are only checked for existence.
const DECODE_LIMIT: usize = 50;

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
        let valid_annotations_path = std::env::var_os("VALID_ANNOTATIONS_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join(dataset::VALID_ANNOTATIONS_PATH)
            });
        let valid_images_root = std::env::var_os("VALID_IMAGES_ROOT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join(dataset::validation_images_root())
            });
        let restart =
            training_restart::TrainingRestart::parse(std::env::args_os().skip(1), EPOCHS)?;
        restart.prepare_run_dir()?;
        let failure = training_failure::TrainingFailure::install(&restart.run_dir)?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            train_dataset(
                &annotations_path,
                &images_root_path,
                &valid_annotations_path,
                &valid_images_root,
                &failure,
                &restart,
            )
        }));
        // Le dashboard a été détruit : ce message reste visible après la restauration du terminal.
        match result {
            Ok(Ok(())) => (),
            Ok(Err(error)) => {
                failure.record_error(&error);
                eprintln!("Rapport d'erreur : {}", failure.path().display());
                return Err(error);
            }
            Err(_) => return Err(format!(
                "Panique pendant l'entraînement. Cause, lot et backtrace : {}. Le dernier checkpoint est conservé.",
                failure.path().display(),
            ).into()),
        }
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
    valid_annotations_path: &std::path::Path,
    valid_images_root: &std::path::Path,
    failure: &training_failure::TrainingFailure,
    restart: &training_restart::TrainingRestart,
) -> Result<(), Box<dyn std::error::Error>> {
    let samples = dataset::load_training_samples(annotations_path, images_root)?;
    let valid_samples =
        dataset::load_validation_samples(valid_annotations_path, valid_images_root)?;
    let device = default_device().autodiff();
    let valid_device = device.clone().inner();
    println!("Using device: {:?}", device);
    println!(
        "Train : {} images ; Valid : {} images",
        samples.len(),
        valid_samples.len()
    );
    device.seed(42);
    println!(
        "Résolution : {}×{} ; grille : {}×{} ; lot : {} ; dossier : {}",
        restart.config.image_size(),
        restart.config.image_size(),
        restart.config.grid_size(),
        restart.config.grid_size(),
        restart.batch_size,
        restart.run_dir.display()
    );

    let mut model = Model::default(&device);
    if let Some(path) = &restart.weights {
        failure.context(format!("Chargement des poids : {}", path.display()));
        model = model.try_load_record(burn::store::ModuleRecord::load(path)?)?;
        println!(
            "Poids chargés : {}. Départ à l'époque {} ; Adam réinitialisé (état non sauvegardé).",
            path.display(),
            restart.first_epoch + 1,
        );
    }
    let mut optimizer = AdamConfig::new().init();

    let batches = samples.len().div_ceil(restart.batch_size);
    let total_steps = EPOCHS * batches;
    let mut dashboard = training_ui::TrainingDashboard::new(
        &restart.run_dir,
        EPOCHS,
        samples.len(),
        restart.first_epoch,
    )?;
    for epoch in restart.first_epoch..EPOCHS {
        dashboard.start_epoch(epoch, samples.len());
        let mut updated = false;
        for (batch_index, entries) in samples.chunks(restart.batch_size).enumerate() {
            if dashboard.should_stop() {
                break;
            }
            failure.context(batch_context(epoch, "Train", batch_index, entries));
            let batch = make_batch(entries, images_root, &device, restart.config)?;

            let predictions = model.forward(batch.images.clone());
            let loss = detection_loss(predictions, &batch);
            let lr = learning_rate(epoch * batches + batch_index, total_steps);

            let diagnostics = loss.diagnostics(&batch)?;
            let gradients = loss.total.backward();
            let gradients = GradientsParams::from_grads(gradients, &model);
            model = optimizer.step(lr, model, gradients);
            updated = true;
            dashboard.update_train(&diagnostics, entries.len(), batch.collisions, lr)?;
        }
        dashboard.end_split();
        if updated {
            failure.context(format!("Époque {} · sauvegarde du checkpoint", epoch + 1));
            model
                .clone()
                .into_record()
                .save(restart.run_dir.join("model.next.bpk"))?;
            // Publier uniquement un checkpoint complètement écrit.
            std::fs::rename(
                restart.run_dir.join("model.next.bpk"),
                restart.run_dir.join("model.bpk"),
            )?;
        }
        if dashboard.should_stop() {
            break;
        }

        // Une vue d'inférence des poids courants ; le modèle d'entraînement reste intact.
        let valid_model = model.valid();
        dashboard.start_validation(valid_samples.len());
        for (batch_index, entries) in valid_samples.chunks(restart.batch_size).enumerate() {
            if dashboard.should_stop() {
                break;
            }
            failure.context(batch_context(epoch, "Valid", batch_index, entries));
            let batch = make_batch(entries, valid_images_root, &valid_device, restart.config)?;
            let diagnostics = validation_diagnostics(&valid_model, &batch)?;
            dashboard.update_valid(&diagnostics, entries.len(), batch.collisions)?;
        }
        dashboard.end_split();
        if dashboard.should_stop() {
            break;
        }
    }
    let interrupted = dashboard.should_stop();
    dashboard.finish()?;
    println!(
        "Entraînement {}. Poids : {} ; diagnostics : {}",
        if interrupted {
            "interrompu"
        } else {
            "terminé"
        },
        restart.run_dir.join("model.bpk").display(),
        restart.run_dir.join("training.log").display(),
    );

    Ok(())
}

fn batch_context(
    epoch: usize,
    split: &str,
    batch: usize,
    entries: &[dataset::AnnotatedImage],
) -> String {
    format!(
        "Époque {} · {split} · lot {}\nImages : {}",
        epoch + 1,
        batch + 1,
        entries
            .iter()
            .map(|entry| entry.image_file_path.as_str())
            .collect::<Vec<_>>()
            .join(", "),
    )
}

fn validation_diagnostics(
    model: &Model,
    batch: &batch::DetectionBatch,
) -> Result<loss::DetectionDiagnostics, Box<dyn std::error::Error>> {
    let predictions = model.forward(batch.images.clone());
    debug_assert!(
        !predictions.is_autodiff(),
        "La validation ne doit pas construire de graphe de gradients"
    );
    detection_loss(predictions, batch).diagnostics(batch)
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
    use burn::module::{ModuleVisitor, Param};
    use burn::tensor::Tensor;

    #[test]
    fn learning_rate_decays_from_max_to_min() {
        assert_eq!(learning_rate(0, 100), LR_MAX);
        assert!((learning_rate(50, 100) - (LR_MAX + LR_MIN) / 2.0).abs() < 1e-12);
        assert!((learning_rate(100, 100) - LR_MIN).abs() < 1e-12);
        assert!(learning_rate(30, 100) > learning_rate(31, 100));
    }

    /// Diagnostic manuel, sans modifier les poids ni le journal d'entraînement.
    #[test]
    #[ignore = "nécessite CHECKPOINT_TO_VALIDATE et le dataset WIDER_val local"]
    fn validate_saved_checkpoint() -> Result<(), Box<dyn std::error::Error>> {
        let path = std::env::var("CHECKPOINT_TO_VALIDATE")?;
        let model = infer::load(std::path::Path::new(&path))?;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let images = root.join(dataset::validation_images_root());
        let samples =
            dataset::load_validation_samples(&root.join(dataset::VALID_ANNOTATIONS_PATH), &images)?;
        let device = default_device();
        for (index, entries) in samples.chunks(4).enumerate() {
            let batch = make_batch(entries, &images, &device, model.config)?;
            let diagnostics = validation_diagnostics(&model.network, &batch)?;
            assert!(diagnostics.total.is_finite(), "lot {}", index + 1);
            if index % 100 == 0 || (590..605).contains(&index) {
                println!("Validation checkpoint · lot {} : {diagnostics}", index + 1);
            }
        }
        println!("Checkpoint valide : {} images évaluées", samples.len());
        Ok(())
    }

    #[derive(Default)]
    struct Parameters {
        values: Vec<Vec<f32>>,
        autodiff: Vec<bool>,
        requires_grad: Vec<bool>,
    }

    impl ModuleVisitor for Parameters {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            let tensor = param.val();
            self.autodiff.push(tensor.is_autodiff());
            self.requires_grad.push(tensor.is_require_grad());
            self.values
                .push(tensor.to_data().try_to_vec::<f32>().unwrap());
        }
    }

    #[test]
    fn validation_preserves_training_parameters_and_disables_autodiff()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        image::RgbImage::from_pixel(160, 80, image::Rgb([255, 128, 0]))
            .save(dir.path().join("sample.png"))?;
        let device = default_device().autodiff();
        device.seed(42);
        let model = Model::default(&device);
        let checkpoint = dir.path().join("model.bpk");
        model.clone().into_record().save(&checkpoint)?;
        let model = Model::default(&device)
            .try_load_record(burn::store::ModuleRecord::load(&checkpoint)?)?;
        let mut before = Parameters::default();
        model.visit(&mut before);
        assert!(!before.values.is_empty());
        assert!(before.autodiff.iter().all(|enabled| *enabled));
        assert!(before.requires_grad.iter().all(|enabled| *enabled));

        let valid_device = device.inner();
        let valid_model = model.valid();
        let mut valid_parameters = Parameters::default();
        valid_model.visit(&mut valid_parameters);
        assert!(valid_parameters.autodiff.iter().all(|enabled| !enabled));
        assert!(
            valid_parameters
                .requires_grad
                .iter()
                .all(|enabled| !enabled)
        );
        assert_eq!(before.values, valid_parameters.values);
        let batch = make_batch(
            &[dataset::AnnotatedImage {
                image_file_path: "sample.png".into(),
                faces: vec![],
            }],
            dir.path(),
            &valid_device,
            detection_config::DetectionConfig::default(),
        )?;
        let diagnostics = validation_diagnostics(&valid_model, &batch)?;
        assert!(diagnostics.total.is_finite());
        assert!(diagnostics.presence.is_finite());
        assert_eq!(diagnostics.localization, 0.0);
        assert_eq!(diagnostics.positives, 0);

        let mut after = Parameters::default();
        model.visit(&mut after);
        assert_eq!(before.values, after.values);
        assert_eq!(before.autodiff, after.autodiff);
        assert_eq!(before.requires_grad, after.requires_grad);
        Ok(())
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
