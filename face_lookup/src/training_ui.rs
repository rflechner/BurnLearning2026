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

struct ScalarMetric {
    definition: MetricDefinition,
    state: NumericMetricState,
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
            state: NumericMetricState::new(),
        }
    }

    fn update(&mut self, value: f64, count: usize) -> MetricState {
        self.state.update(value, count);
        let entry = self
            .state
            .compute_update(FormatOptions::new(self.definition.name.clone()).precision(6));
        MetricState::Numeric(
            MetricEntry::new(self.definition.metric_id.clone(), entry),
            Some(self.state.current_value()),
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
    pub fn new(run_dir: &Path, epochs: usize, items: usize) -> io::Result<Self> {
        // Ouvrir le journal avant de prendre le contrôle du terminal.
        let log = File::create(run_dir.join("training.log"))?;
        let interrupter = Interrupter::new();
        let terminal = io::stdout().is_terminal();
        let renderer = default_renderer(interrupter.clone(), None);
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
            metric.state.reset();
        }
        // Burn attend une époque à partir de 1 pour son calcul de progression.
        self.renderer.update_epoch(epoch);
        self.renderer.start_split("train", items);
        self.renderer.update_split(0);
    }

    pub fn update(
        &mut self,
        diagnostics: &DetectionDiagnostics,
        batch_size: usize,
        collisions: usize,
        lr: f64,
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
            self.renderer
                .update_train(metric.update(value as f64, diagnostics.positives.max(1)));
        }
        self.renderer.update_train(self.learning_rate.update(lr, 1));
        self.renderer.update_split(self.items_processed);

        let line = format!(
            "Époque {} · lot {} : {diagnostics}, collisions={collisions}, lr={lr:.2e}",
            self.epoch, self.batch,
        );
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

    pub fn end_epoch(&mut self) {
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
        dashboard.update(&diagnostics, 4, 2, 1e-3)?;
        // Lot final incomplet et sans positif : dénominateur borné à 1, comme la loss.
        diagnostics.total = 1.0;
        diagnostics.presence = 1.0;
        diagnostics.localization = 0.0;
        diagnostics.positives = 0;
        dashboard.update(&diagnostics, 1, 0, 1e-4)?;
        assert_eq!(dashboard.items_processed, 5);
        let averages: Vec<_> = dashboard
            .losses
            .iter()
            .map(|metric| metric.state.running_value().current())
            .collect();
        assert_eq!(
            averages,
            vec![4.0, 1.75, 2.25, 2.25 * LOCALIZATION_WEIGHT as f64]
        );
        dashboard.end_epoch();
        dashboard.start_epoch(1, 5);
        dashboard.update(&diagnostics, 1, 0, 1e-5)?;
        assert_eq!(dashboard.losses[0].state.running_value().current(), 1.0);
        assert_eq!(dashboard.items_processed, 1);
        dashboard.interrupter.stop(Some("test"));
        assert!(dashboard.should_stop());
        dashboard.end_epoch();
        dashboard.finish()?;
        let log = std::fs::read_to_string(dir.path().join("training.log"))?;
        assert_eq!(log.lines().count(), 3);
        assert!(log.contains(
            "Époque 1 · lot 1 : perte=5.000000, présence=2.000000, localisation=3.000000"
        ));
        assert!(log.contains("collisions=2"));
        assert!(log.contains("Époque 2 · lot 1 : perte=1.000000"));
        Ok(())
    }
}
