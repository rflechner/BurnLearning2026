use crate::batch::DetectionBatch;
use burn::tensor::{
    Tensor,
    activation::{sigmoid, softplus},
};

/// Présence : BCE stable sur logits, positifs et négatifs normalisés séparément.
/// Boîtes : Smooth L1 sur les seules cellules positives, après décodage.
pub fn detection_loss(predictions: Tensor<4>, batch: &DetectionBatch) -> DetectionLoss {
    let [n, _, h, w] = predictions.dims();
    let logits = predictions.clone().slice([0..n, 0..1, 0..h, 0..w]);
    let bce = logits.clone().clamp_min(0.0) - logits.clone() * batch.presence.clone()
        + ((-logits.abs()).exp() + 1.0).log();
    let positive = batch.positive_mask.clone();
    let negative = (batch.presence.clone().neg() + 1.0) * batch.valid_mask.clone();
    let positive_count = positive.clone().sum().clamp_min(1.0);
    let presence_loss = (bce.clone() * positive.clone()).sum() / positive_count.clone()
        + (bce * negative.clone()).sum() / negative.sum().clamp_min(1.0);
    let offsets = sigmoid(predictions.clone().slice([0..n, 1..3, 0..h, 0..w]));
    let sizes = softplus(predictions.slice([0..n, 3..5, 0..h, 0..w]), 1.0);
    let error = (Tensor::cat(vec![offsets, sizes.clone()], 1) - batch.boxes.clone()).abs();
    let quadratic = error.clone().clamp_max(1.0);
    let smooth_l1 = quadratic.clone() * quadratic.clone() * 0.5 + error - quadratic;
    let box_loss = (smooth_l1 * positive).sum() / (positive_count * 4.0);
    DetectionLoss {
        total: presence_loss.clone() + box_loss.clone() * LOCALIZATION_WEIGHT,
        presence: presence_loss,
        localization: box_loss,
        predicted_sizes: sizes,
    }
}

pub const LOCALIZATION_WEIGHT: f32 = 5.0;

pub struct DetectionLoss {
    pub total: Tensor<1>,
    pub presence: Tensor<1>,
    /// Perte brute : sa contribution au total est multipliée par 5.
    pub localization: Tensor<1>,
    predicted_sizes: Tensor<4>,
}

impl DetectionLoss {
    /// Mesures avant la mise à jour, aux seules cellules positives.
    /// Les tailles sont en pixels de l'entrée 320×320, pas de l'image originale.
    pub fn diagnostics(
        &self,
        batch: &DetectionBatch,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let total = self.total.clone().into_scalar::<f32>();
        let presence = self.presence.clone().into_scalar::<f32>();
        let localization = self.localization.clone().into_scalar::<f32>();
        // Lire des copies pour les logs sans modifier le graphe utilisé par backward().
        let predicted = self.predicted_sizes.to_data().try_to_vec::<f32>()?;
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
                    let p = predicted[(image * 2 + channel) * cells + cell] as f64
                        * crate::batch::IMAGE_SIZE as f64;
                    let t = expected[(image * 4 + channel + 2) * cells + cell] as f64
                        * crate::batch::IMAGE_SIZE as f64;
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
        Ok(format!(
            "perte={total:.6}, présence={presence:.6}, localisation={localization:.6}, localisation_pondérée={:.6}, positifs={count}, {sizes}",
            localization * LOCALIZATION_WEIGHT
        ))
    }
}
