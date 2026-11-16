//! Résolution commune à l'entraînement et à l'inférence, conservée avec les poids.
use serde::{Deserialize, Serialize};
use std::{io, path::Path};

pub const STRIDE: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetectionConfig {
    image_size: usize,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self { image_size: 320 }
    }
}

impl DetectionConfig {
    pub fn new(image_size: usize) -> io::Result<Self> {
        if !(320..=1280).contains(&image_size) || image_size % 32 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "La résolution doit être un multiple de 32 entre 320 et 1280 (par exemple 640)",
            ));
        }
        Ok(Self { image_size })
    }

    pub fn image_size(self) -> usize {
        self.image_size
    }

    pub fn grid_size(self) -> usize {
        self.image_size / STRIDE
    }

    /// Les checkpoints historiques sans configuration ont été entraînés en 320 px.
    pub fn for_weights(path: &Path) -> io::Result<Self> {
        match std::fs::read(path.with_extension("config.json")) {
            Ok(bytes) => {
                let config: Self = serde_json::from_slice(&bytes)
                    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
                Self::new(config.image_size)
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Self::default()),
            Err(error) => Err(error),
        }
    }

    /// Configuration immuable dans un dossier d'entraînement, publiée avant les poids.
    pub fn save_for_weights(self, path: &Path) -> io::Result<()> {
        let config_path = path.with_extension("config.json");
        if config_path.exists() {
            if Self::for_weights(path)? != self {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Ce dossier utilise une autre résolution ; choisissez un autre --run-dir",
                ));
            }
            return Ok(());
        }
        let bytes = serde_json::to_vec_pretty(&self).map_err(io::Error::other)?;
        let temporary = path.with_extension("config.json.tmp");
        std::fs::write(&temporary, bytes)?;
        std::fs::rename(temporary, config_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_resolution_roundtrip_and_legacy_default() -> io::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("model.bpk");
        assert_eq!(DetectionConfig::for_weights(&path)?.image_size(), 320);
        let config = DetectionConfig::new(640)?;
        config.save_for_weights(&path)?;
        assert_eq!(DetectionConfig::for_weights(&path)?, config);
        assert_eq!(config.grid_size(), 80);
        assert!(DetectionConfig::default().save_for_weights(&path).is_err());
        std::fs::write(path.with_extension("config.json"), "{\"image_size\":0}")?;
        assert!(DetectionConfig::for_weights(&path).is_err());
        for size in [0, 319, 321, 1281, usize::MAX] {
            assert!(DetectionConfig::new(size).is_err());
        }
        Ok(())
    }
}
