use std::{ffi::OsString, io, path::PathBuf};

/// Repartir de poids sauvegardés ; les moments d'Adam ne sont pas dans model.bpk.
#[derive(Default)]
pub struct TrainingRestart {
    pub weights: Option<PathBuf>,
    pub first_epoch: usize,
}

impl TrainingRestart {
    pub fn parse(args: impl IntoIterator<Item = OsString>, epochs: usize) -> io::Result<Self> {
        let mut args = args.into_iter();
        let mut options = Self::default();
        let mut start_epoch = None;
        while let Some(arg) = args.next() {
            if arg == "--resume-weights" {
                options.weights =
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
            }
        }
        if let Some(epoch) = start_epoch {
            if options.weights.is_none() {
                return Err(invalid("--start-epoch nécessite --resume-weights"));
            }
            if !(1..=epochs).contains(&epoch) {
                return Err(invalid(&format!(
                    "--start-epoch doit être compris entre 1 et {epochs}"
                )));
            }
            options.first_epoch = epoch - 1;
        }
        Ok(options)
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;

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
