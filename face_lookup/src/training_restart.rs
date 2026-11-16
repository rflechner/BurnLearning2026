use crate::detection_config::DetectionConfig;
use std::{ffi::OsString, io, path::PathBuf};

/// Repartir de poids sauvegardés ; les moments d'Adam ne sont pas dans model.bpk.
pub struct TrainingRestart {
    pub weights: Option<PathBuf>,
    pub first_epoch: usize,
    pub config: DetectionConfig,
    pub run_dir: PathBuf,
    pub batch_size: usize,
}

impl TrainingRestart {
    pub fn parse(args: impl IntoIterator<Item = OsString>, epochs: usize) -> io::Result<Self> {
        let mut args = args.into_iter();
        let mut weights = None;
        let mut start_epoch = None;
        let mut image_size = None;
        let mut run_dir = None;
        let mut batch_size = None;
        while let Some(arg) = args.next() {
            if arg == "--resume-weights" {
                weights =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        invalid("Chemin manquant après --resume-weights")
                    })?));
            } else if arg == "--start-epoch" {
                let value = args
                    .next()
                    .ok_or_else(|| invalid("Numéro manquant après --start-epoch"))?;
                start_epoch = Some(
                    value
                        .to_str()
                        .and_then(|s| s.parse::<usize>().ok())
                        .ok_or_else(|| invalid("--start-epoch attend un entier"))?,
                );
            } else if arg == "--image-size" || arg == "--batch-size" {
                let value = args
                    .next()
                    .and_then(|s| s.to_str()?.parse::<usize>().ok())
                    .ok_or_else(|| {
                        invalid(&format!(
                            "{} attend un entier positif",
                            arg.to_string_lossy()
                        ))
                    })?;
                if arg == "--image-size" {
                    image_size = Some(DetectionConfig::new(value)?);
                } else if value == 0 {
                    return Err(invalid("--batch-size doit être supérieur à zéro"));
                } else {
                    batch_size = Some(value);
                }
            } else if arg == "--run-dir" {
                let value = args
                    .next()
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| invalid("Dossier manquant après --run-dir"))?;
                run_dir = Some(PathBuf::from(value));
            } else if arg != "--train" {
                return Err(invalid(&format!(
                    "Option inconnue : {}",
                    arg.to_string_lossy()
                )));
            }
        }
        let mut first_epoch = 0;
        if let Some(epoch) = start_epoch {
            if weights.is_none() {
                return Err(invalid("--start-epoch nécessite --resume-weights"));
            }
            if !(1..=epochs).contains(&epoch) {
                return Err(invalid(&format!(
                    "--start-epoch doit être compris entre 1 et {epochs}"
                )));
            }
            first_epoch = epoch - 1;
        }
        let config = match image_size {
            Some(config) => config,
            None => weights
                .as_deref()
                .map(DetectionConfig::for_weights)
                .transpose()?
                .unwrap_or_default(),
        };
        let run_dir = run_dir.unwrap_or_else(|| {
            if config.image_size() == 320 {
                PathBuf::from("runs/dilated")
            } else {
                PathBuf::from(format!("runs/dilated-{}", config.image_size()))
            }
        });
        Ok(Self {
            weights,
            first_epoch,
            config,
            run_dir,
            batch_size: batch_size.unwrap_or(if config.image_size() == 320 { 4 } else { 1 }),
        })
    }

    /// Vérifier la destination avant de recréer les journaux ou de lancer le GPU.
    pub fn prepare_run_dir(&self) -> io::Result<()> {
        if let Some(source) = &self.weights {
            if !std::fs::metadata(source)?.is_file() {
                return Err(invalid("--resume-weights doit désigner un fichier"));
            }
        }
        let target = self.run_dir.join("model.bpk");
        if target.exists() {
            if DetectionConfig::for_weights(&target)? != self.config {
                return Err(invalid(
                    "Les poids de ce dossier utilisent une autre résolution ; choisissez un autre --run-dir",
                ));
            }
            let source = self
                .weights
                .as_deref()
                .map(std::fs::canonicalize)
                .transpose()?;
            if source.as_ref() != Some(&std::fs::canonicalize(&target)?) {
                return Err(invalid(
                    "Ce dossier contient déjà model.bpk ; utilisez un nouveau --run-dir ou reprenez ce checkpoint avec --resume-weights",
                ));
            }
        }
        std::fs::create_dir_all(&self.run_dir)?;
        self.config.save_for_weights(&target)
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolution_options_and_destination_protect_existing_weights() -> io::Result<()> {
        let parse = |args: &[&str]| TrainingRestart::parse(args.iter().map(OsString::from), 100);
        let standard = parse(&["--train"])?;
        assert_eq!(standard.config.image_size(), 320);
        assert_eq!(standard.batch_size, 4);
        let mut high = parse(&["--train", "--image-size", "640"])?;
        assert_eq!(high.config.image_size(), 640);
        assert_eq!(high.batch_size, 1);
        assert_eq!(high.run_dir, PathBuf::from("runs/dilated-640"));
        assert_eq!(
            parse(&["--batch-size", "2", "--run-dir", "runs/experiment"])?.batch_size,
            2
        );
        for args in [
            vec!["--image-size", "321"],
            vec!["--image-size"],
            vec!["--batch-size", "0"],
            vec!["--run-dir"],
            vec!["--typo"],
        ] {
            assert!(parse(&args).is_err());
        }
        let dir = tempfile::tempdir()?;
        let legacy = dir.path().join("legacy");
        std::fs::create_dir(&legacy)?;
        let legacy_weights = legacy.join("model.bpk");
        std::fs::write(&legacy_weights, b"original weights")?;
        high.run_dir = legacy;
        high.weights = Some(legacy_weights.clone());
        assert!(high.prepare_run_dir().is_err());
        assert!(!legacy_weights.with_extension("config.json").exists());
        high.run_dir = dir.path().join("high");
        high.prepare_run_dir()?;
        assert_eq!(std::fs::read(&legacy_weights)?, b"original weights");
        let target = high.run_dir.join("model.bpk");
        assert!(!target.exists());
        assert_eq!(DetectionConfig::for_weights(&target)?.image_size(), 640);
        std::fs::write(&target, b"new weights")?;
        assert!(high.prepare_run_dir().is_err());
        high.weights = Some(target.clone());
        high.prepare_run_dir()?;
        let resumed = TrainingRestart::parse(
            [OsString::from("--resume-weights"), target.into_os_string()],
            100,
        )?;
        assert_eq!(resumed.config.image_size(), 640);
        Ok(())
    }

    #[test]
    fn restart_requires_weights_and_a_valid_epoch() {
        let parse = |args: &[&str]| TrainingRestart::parse(args.iter().map(OsString::from), 100);
        let restart = parse(&[
            "--train",
            "--resume-weights",
            "saved model.bpk",
            "--start-epoch",
            "7",
        ])
        .unwrap();
        assert_eq!(restart.weights.unwrap(), PathBuf::from("saved model.bpk"));
        assert_eq!(restart.first_epoch, 6);
        assert!(parse(&["--start-epoch", "7"]).is_err());
        assert!(parse(&["--resume-weights"]).is_err());
        assert!(parse(&["--resume-weights", "model.bpk", "--start-epoch", "0"]).is_err());
        assert!(parse(&["--resume-weights", "model.bpk", "--start-epoch", "101"]).is_err());
        assert_eq!(parse(&["--train"]).unwrap().first_epoch, 0);
    }
}
