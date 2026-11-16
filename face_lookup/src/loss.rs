use crate::batch::DetectionBatch;
use burn::tensor::{
    Tensor,
    activation::{log_sigmoid, sigmoid},
};

/// Présence : focal loss façon CenterNet, normalisée par le nombre de positifs.
/// Le facteur p² rend négligeables les milliers de cellules de fond déjà bien classées,
/// et (1 − gaussienne)⁴ allège la pénalité des voisines d'un centre.
/// Boîtes : L1 sur les seules cellules positives, gradient constant même pour une petite erreur.
pub fn detection_loss(predictions: Tensor<4>, batch: &DetectionBatch) -> DetectionLoss {
    let [n, _, h, w] = predictions.dims();
    let logits = predictions.clone().slice([0..n, 0..1, 0..h, 0..w]);
    let score = sigmoid(logits.clone());
    // log(p) et log(1 − p) calculés depuis les logits, stables même quand p arrondit à 0 ou 1.
    let log_score = log_sigmoid(logits.clone());
    let log_background = log_sigmoid(logits.neg());
    let positive = batch.positive_mask.clone();
    let positive_count = positive.clone().sum().clamp_min(1.0);
    let positive_loss =
        (score.clone().neg() + 1.0).powi_scalar(2) * log_score.neg() * positive.clone();
    let negative_weight = (batch.heatmap.clone().neg() + 1.0).powi_scalar(4)
        * (positive.clone().neg() + 1.0)
        * batch.valid_mask.clone();
    let negative_loss = score.powi_scalar(2) * log_background.neg() * negative_weight;
    let presence_loss = (positive_loss.sum() + negative_loss.sum()) / positive_count.clone();
    let offsets = sigmoid(predictions.clone().slice([0..n, 1..3, 0..h, 0..w]));
    let log_sizes = predictions.slice([0..n, 3..5, 0..h, 0..w]);
    let error = (Tensor::cat(vec![offsets, log_sizes.clone()], 1) - batch.boxes.clone()).abs();
    let box_loss = (error * positive).sum() / (positive_count * 4.0);
    DetectionLoss {
        total: presence_loss.clone() + box_loss.clone() * LOCALIZATION_WEIGHT,
        presence: presence_loss,
        localization: box_loss,
        predicted_log_sizes: log_sizes,
    }
}

pub const LOCALIZATION_WEIGHT: f32 = 1.0;

pub struct DetectionLoss {
    pub total: Tensor<1>,
    pub presence: Tensor<1>,
    /// Perte brute : sa contribution au total est multipliée par `LOCALIZATION_WEIGHT`.
    pub localization: Tensor<1>,
    predicted_log_sizes: Tensor<4>,
}

/// Valeurs détachées du graphe, partagées par le tableau de bord et le journal.
pub struct DetectionDiagnostics {
    pub total: f32,
    pub presence: f32,
    pub localization: f32,
    pub positives: usize,
    pub sizes: String,
}

impl std::fmt::Display for DetectionDiagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "perte={:.6}, présence={:.6}, localisation={:.6}, localisation_pondérée={:.6}, positifs={}, {}",
            self.total,
            self.presence,
            self.localization,
            self.localization * LOCALIZATION_WEIGHT,
            self.positives,
            self.sizes,
        )
    }
}

impl DetectionLoss {
    /// Mesures avant la mise à jour, aux seules cellules positives.
    /// Les tailles sont en pixels de l'entrée 320×320, pas de l'image originale.
    pub fn diagnostics(
        &self,
        batch: &DetectionBatch,
    ) -> Result<DetectionDiagnostics, Box<dyn std::error::Error>> {
        let total = self.total.clone().into_scalar::<f32>();
        let presence = self.presence.clone().into_scalar::<f32>();
        let localization = self.localization.clone().into_scalar::<f32>();
        // Lire des copies pour les logs sans modifier le graphe utilisé par backward().
        let predicted = self.predicted_log_sizes.to_data().try_to_vec::<f32>()?;
        let expected = batch.boxes.to_data().try_to_vec::<f32>()?;
        let mask = batch.positive_mask.to_data().try_to_vec::<f32>()?;
        let [n, _, h, w] = batch.boxes.dims();
        let cells = h * w;
        let mut count = 0;
        let mut predicted_sum = [0.0f64; 2];
        let mut expected_sum = [0.0f64; 2];
        let mut absolute_error = [0.0f64; 2];
        for image in 0..n {
            for cell in 0..cells {
                if mask[image * cells + cell] <= 0.0 {
                    continue;
                }
                count += 1;
                for channel in 0..2 {
                    // Repasser des log-tailles aux pixels ; borner pour un affichage fini.
                    let to_pixels =
                        |v: f32| (v as f64).min(20.0).exp() * crate::batch::STRIDE as f64;
                    let p = to_pixels(predicted[(image * 2 + channel) * cells + cell]);
                    let t = to_pixels(expected[(image * 4 + channel + 2) * cells + cell]);
                    predicted_sum[channel] += p;
                    expected_sum[channel] += t;
                    absolute_error[channel] += (p - t).abs();
                }
            }
        }
        let sizes = if count == 0 {
            "tailles=n/a (aucun positif)".to_string()
        } else {
            let n = count as f64;
            format!(
                "taille_moyenne_predite={:.2}×{:.2}px, attendue={:.2}×{:.2}px, erreur_absolue_moyenne={:.2}×{:.2}px (L×H, entrée 320×320)",
                predicted_sum[0] / n,
                predicted_sum[1] / n,
                expected_sum[0] / n,
                expected_sum[1] / n,
                absolute_error[0] / n,
                absolute_error[1] / n
            )
        };
        Ok(DetectionDiagnostics {
            total,
            presence,
            localization,
            positives: count,
            sizes,
        })
    }
}
