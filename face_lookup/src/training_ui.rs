//! Tableau de bord natif de Burn pour notre boucle d'entraînement personnalisée.

use crate::loss::{DetectionDiagnostics, LOCALIZATION_WEIGHT};
use burn::train::{
    Interrupter,
    metric::{
        MetricAttributes, MetricDefinition, MetricEntry, MetricId, NumericAttributes,
        state::{FormatOptions, NumericMetricState},
    },
    renderer::{MetricState, MetricsRenderer, default_renderer},
};
use std::{
    fs::File,
    io::{self, BufWriter, IsTerminal, Write},
    path::Path,
    sync::Arc,
};

#[derive(Clone, Copy)]
enum Split {
    Train,
    Valid,
}

impl Split {
    fn label(self) -> &'static str {
        match self {
            Self::Train => "Train",
            Self::Valid => "Valid",
        }
    }
}

struct ScalarMetric {
    definition: MetricDefinition,
    states: [NumericMetricState; 2],
}

impl ScalarMetric {
    fn new(name: &str, description: &str) -> Self {
        Self {
            definition: definition(
                name,
                description,
                NumericAttributes {
                    unit: None,
                    higher_is_better: false,
                }
                .into(),
            ),
            states: std::array::from_fn(|_| NumericMetricState::new()),
        }
    }

    fn update(&mut self, split: Split, value: f64, count: usize) -> MetricState {
        let state = &mut self.states[split as usize];
        state.update(value, count);
        let entry =
            state.compute_update(FormatOptions::new(self.definition.name.clone()).precision(6));
        MetricState::Numeric(
            MetricEntry::new(self.definition.metric_id.clone(), entry),
            Some(state.current_value()),
        )
    }
}

fn definition(name: &str, description: &str, attributes: MetricAttributes) -> MetricDefinition {
    let name = Arc::new(name.to_string());
    MetricDefinition {
        metric_id: MetricId::new(name.clone()),
        name,
        description: Some(description.into()),
        attributes,
    }
}

pub struct TrainingDashboard {
    renderer: Box<dyn MetricsRenderer>,
    interrupter: Interrupter,
    losses: [ScalarMetric; 4],
    learning_rate: ScalarMetric,
    log: BufWriter<File>,
    terminal: bool,
    epoch: usize,
    batch: usize,
    items_processed: usize,
}

impl TrainingDashboard {
    pub fn new(
        run_dir: &Path,
        epochs: usize,
        items: usize,
        first_epoch: usize,
    ) -> io::Result<Self> {
        // Ouvrir le journal avant de prendre le contrôle du terminal.
        let log = File::create(run_dir.join("training.log"))?;
        let interrupter = Interrupter::new();
        let terminal = io::stdout().is_terminal();
        let renderer = default_renderer(interrupter.clone(), Some(first_epoch));
        Ok(Self::with_renderer(
            log,
            epochs,
            items,
            renderer,
            interrupter,
            terminal,
        ))
    }

    fn with_renderer(
        log: File,
        epochs: usize,
        items: usize,
        mut renderer: Box<dyn MetricsRenderer>,
        interrupter: Interrupter,
        terminal: bool,
    ) -> Self {
        let losses = [
            ScalarMetric::new("Loss totale", "Présence + localisation pondérée"),
            ScalarMetric::new("Loss présence", "Focal loss de présence des visages"),
            ScalarMetric::new("Loss localisation", "L1 brute des offsets et log-tailles"),
            ScalarMetric::new("Loss localisation pondérée", "L1 × LOCALIZATION_WEIGHT"),
        ];
        let learning_rate = ScalarMetric::new("Learning rate", "Taux d'apprentissage cosinus");
        for metric in losses.iter().chain(std::iter::once(&learning_rate)) {
            renderer.register_metric(metric.definition.clone());
        }
        renderer.start(epochs, 1, Some(items), Some("Détection de visages"));
        Self {
            renderer,
            interrupter,
            losses,
            learning_rate,
            log: BufWriter::new(log),
            terminal,
            epoch: 0,
            batch: 0,
            items_processed: 0,
        }
    }

    pub fn start_epoch(&mut self, epoch: usize, items: usize) {
        self.epoch = epoch + 1;
        self.batch = 0;
        self.items_processed = 0;
        for metric in self
            .losses
            .iter_mut()
            .chain(std::iter::once(&mut self.learning_rate))
        {
            for state in &mut metric.states {
                state.reset();
            }
        }
        // update_epoch attend un index à partir de 0 et affiche epoch + 1.
        self.renderer.update_epoch(epoch);
        self.renderer.start_split("train", items);
        self.renderer.update_split(0);
    }

    pub fn start_validation(&mut self, items: usize) {
        self.batch = 0;
        self.items_processed = 0;
        self.renderer.start_split("valid", items);
        self.renderer.update_split(0);
    }

    pub fn update_train(
        &mut self,
        diagnostics: &DetectionDiagnostics,
        batch_size: usize,
        collisions: usize,
        lr: f64,
    ) -> io::Result<()> {
        self.update(Split::Train, diagnostics, batch_size, collisions, Some(lr))
    }

    pub fn update_valid(
        &mut self,
        diagnostics: &DetectionDiagnostics,
        batch_size: usize,
        collisions: usize,
    ) -> io::Result<()> {
        self.update(Split::Valid, diagnostics, batch_size, collisions, None)
    }

    fn update(
        &mut self,
        split: Split,
        diagnostics: &DetectionDiagnostics,
        batch_size: usize,
        collisions: usize,
        lr: Option<f64>,
    ) -> io::Result<()> {
        self.batch += 1;
        self.items_processed += batch_size;
        let values = [
            diagnostics.total,
            diagnostics.presence,
            diagnostics.localization,
            diagnostics.localization * LOCALIZATION_WEIGHT,
        ];
        for (metric, value) in self.losses.iter_mut().zip(values) {
            // La loss est normalisée par les positifs, et non par le nombre d'images.
            // Garder le même dénominateur pour agréger correctement les lots d'une époque.
            let state = metric.update(split, value as f64, diagnostics.positives.max(1));
            match split {
                Split::Train => self.renderer.update_train(state),
                Split::Valid => self.renderer.update_valid(state),
            }
        }
        if let Some(lr) = lr {
            self.renderer
                .update_train(self.learning_rate.update(Split::Train, lr, 1));
        }
        self.renderer.update_split(self.items_processed);

        let mut line = format!(
            "Époque {} · {} · lot {} : {diagnostics}, collisions={collisions}",
            self.epoch,
            split.label(),
            self.batch,
        );
        if let Some(lr) = lr {
            line.push_str(&format!(", lr={lr:.2e}"));
        }
        writeln!(self.log, "{line}")?;
        self.log.flush()?;
        // Le renderer CLI de Burn n'imprime que la progression, pas les métriques.
        if !self.terminal {
            println!("{line}");
        }
        Ok(())
    }

    pub fn should_stop(&self) -> bool {
        self.interrupter.should_stop()
    }

    pub fn end_split(&mut self) {
        self.renderer.end_split();
    }

    pub fn finish(mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.log.flush()?;
        self.renderer.end();
        self.renderer.on_train_end(None)?;
        // Le Drop du renderer restaure le terminal, y compris sur une erreur remontée par `?`.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rejoue les valeurs réelles sans GPU, pour isoler les erreurs du renderer.
    #[test]
    #[ignore = "nécessite REPLAY_TRAINING_LOG et un vrai terminal"]
    fn replay_training_log() -> Result<(), Box<dyn std::error::Error>> {
        let source = std::fs::read_to_string(std::env::var("REPLAY_TRAINING_LOG")?)?;
        let dir = tempfile::tempdir()?;
        let mut dashboard = TrainingDashboard::new(dir.path(), 100, 12880, 0)?;
        let mut phase = (0, false);
        for line in source.lines() {
            let parts: Vec<_> = line.split(" · ").collect();
            let epoch: usize = parts[0].strip_prefix("Époque ").unwrap().parse()?;
            let valid = parts[1] == "Valid";
            if phase.0 != epoch {
                if phase.0 != 0 {
                    dashboard.end_split();
                }
                dashboard.start_epoch(epoch - 1, 12880);
            } else if phase.1 != valid {
                dashboard.end_split();
            }
            if valid && phase != (epoch, valid) {
                dashboard.start_validation(3226);
            }
            phase = (epoch, valid);
            let value = |key: &str| -> &str {
                line.split_once(key)
                    .unwrap()
                    .1
                    .split(',')
                    .next()
                    .unwrap()
                    .trim()
            };
            let diagnostics = DetectionDiagnostics {
                total: value("perte=").parse()?,
                presence: value("présence=").parse()?,
                localization: value("localisation=").parse()?,
                positives: value("positifs=").parse()?,
                sizes: "relecture du journal".into(),
            };
            if valid {
                let batch_size = if parts[2].starts_with("lot 807 :") {
                    2
                } else {
                    4
                };
                dashboard.update_valid(&diagnostics, batch_size, 0)?;
            } else {
                dashboard.update_train(&diagnostics, 4, 0, value("lr=").parse()?)?;
            }
        }
        dashboard.end_split();
        dashboard.finish()?;
        Ok(())
    }

    #[test]
    fn metrics_preserve_components_and_reset_epoch_averages()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let mut dashboard = TrainingDashboard::with_renderer(
            File::create(dir.path().join("training.log"))?,
            2,
            5,
            Box::new(burn::train::renderer::CliMetricsRenderer::new()),
            Interrupter::new(),
            false,
        );
        dashboard.start_epoch(0, 5);
        let mut diagnostics = DetectionDiagnostics {
            total: 5.0,
            presence: 2.0,
            localization: 3.0,
            positives: 3,
            sizes: "tailles de test".into(),
        };
        dashboard.update_train(&diagnostics, 4, 2, 1e-3)?;
        // Lot final incomplet et sans positif : dénominateur borné à 1, comme la loss.
        diagnostics.total = 1.0;
        diagnostics.presence = 1.0;
        diagnostics.localization = 0.0;
        diagnostics.positives = 0;
        dashboard.update_train(&diagnostics, 1, 0, 1e-4)?;
        assert_eq!(dashboard.items_processed, 5);
        let averages: Vec<_> = dashboard
            .losses
            .iter()
            .map(|metric| {
                metric.states[Split::Train as usize]
                    .running_value()
                    .current()
            })
            .collect();
        assert_eq!(
            averages,
            vec![4.0, 1.75, 2.25, 2.25 * LOCALIZATION_WEIGHT as f64]
        );
        dashboard.end_split();
        dashboard.start_validation(1);
        diagnostics.total = 9.0;
        diagnostics.presence = 8.0;
        diagnostics.localization = 1.0;
        diagnostics.positives = 1;
        dashboard.update_valid(&diagnostics, 1, 0)?;
        assert_eq!(
            dashboard.losses[0].states[Split::Train as usize]
                .running_value()
                .current(),
            4.0
        );
        assert_eq!(
            dashboard.losses[0].states[Split::Valid as usize]
                .running_value()
                .current(),
            9.0
        );
        dashboard.end_split();
        dashboard.start_epoch(1, 5);
        diagnostics.total = 1.0;
        diagnostics.presence = 1.0;
        diagnostics.localization = 0.0;
        diagnostics.positives = 0;
        dashboard.update_train(&diagnostics, 1, 0, 1e-5)?;
        assert_eq!(
            dashboard.losses[0].states[Split::Train as usize]
                .running_value()
                .current(),
            1.0
        );
        assert!(
            dashboard.losses[0].states[Split::Valid as usize]
                .running_value()
                .current()
                .is_nan()
        );
        assert_eq!(dashboard.items_processed, 1);
        dashboard.interrupter.stop(Some("test"));
        assert!(dashboard.should_stop());
        dashboard.end_split();
        dashboard.finish()?;
        let log = std::fs::read_to_string(dir.path().join("training.log"))?;
        assert_eq!(log.lines().count(), 4);
        assert!(log.contains(
            "Époque 1 · Train · lot 1 : perte=5.000000, présence=2.000000, localisation=3.000000"
        ));
        assert!(log.contains("collisions=2"));
        assert!(log.contains("Époque 2 · Train · lot 1 : perte=1.000000"));
        let valid_line = log
            .lines()
            .find(|line| line.contains(" · Valid · "))
            .unwrap();
        assert!(valid_line.contains("perte=9.000000"));
        assert!(!valid_line.contains("lr="));
        Ok(())
    }
}
