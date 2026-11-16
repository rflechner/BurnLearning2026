//! Rapport persistant, même si le terminal alternatif masque une panique.

use burn::train::{ApplicationLoggerInstaller, FileApplicationLoggerInstaller};
use std::{
    backtrace::Backtrace,
    fs::OpenOptions,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::SystemTime,
};

pub struct TrainingFailure {
    path: PathBuf,
    context: Arc<Mutex<String>>,
}

impl TrainingFailure {
    /// À installer avant le renderer Burn, qui chaînera ce hook au sien.
    pub fn install(run_dir: &Path) -> io::Result<Self> {
        std::fs::create_dir_all(run_dir)?;
        let path = run_dir.join("failure.log");
        OpenOptions::new().create(true).append(true).open(&path)?;
        FileApplicationLoggerInstaller::new(run_dir.join("runtime.log"))
            .install()
            .map_err(io::Error::other)?;
        let context = Arc::new(Mutex::new("Initialisation de l'entraînement".to_string()));
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new({
            let path = path.clone();
            let context = context.clone();
            move |info| {
                let context = context.try_lock().map(|c| c.clone()).unwrap_or_default();
                append(
                    &path,
                    &context,
                    &format!("{info}\n{}", Backtrace::force_capture()),
                );
                previous(info);
            }
        }));
        Ok(Self { path, context })
    }

    pub fn context(&self, message: impl Into<String>) {
        if let Ok(mut context) = self.context.lock() {
            *context = message.into();
        }
    }

    pub fn record_error(&self, error: &dyn std::fmt::Display) {
        let context = self.context.lock().map(|c| c.clone()).unwrap_or_default();
        append(&self.path, &context, &error.to_string());
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn append(path: &Path, context: &str, error: &str) {
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "\n{:?}\n{context}\n{error}", SystemTime::now());
        let _ = file.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_is_preserved_with_context_and_backtrace() -> io::Result<()> {
        let dir = tempfile::tempdir()?;
        let child = std::process::Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "training_failure::tests::panic_child",
                "--ignored",
            ])
            .env("PANIC_REPORT_TEST_DIR", dir.path())
            .output()?;
        assert!(!child.status.success());
        let report = std::fs::read_to_string(dir.path().join("failure.log"))?;
        assert!(report.contains("Époque 6 · Valid · lot 597 · sample.jpg"));
        assert!(report.contains("panique de contrôle"));
        assert!(report.contains("panic_child"));
        assert!(
            std::fs::read_to_string(dir.path().join("runtime.log"))?
                .contains("panique de contrôle")
        );
        Ok(())
    }

    #[test]
    #[ignore = "sous-processus du test de rapport de panique"]
    fn panic_child() {
        let Some(dir) = std::env::var_os("PANIC_REPORT_TEST_DIR") else {
            return;
        };
        let report = TrainingFailure::install(Path::new(&dir)).unwrap();
        report.context("Époque 6 · Valid · lot 597 · sample.jpg");
        panic!("panique de contrôle");
    }
}
