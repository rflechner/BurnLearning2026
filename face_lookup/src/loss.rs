use crate::batch::DetectionBatch;
use burn::tensor::{
    Tensor,
    activation::{sigmoid, softplus},
};

/// Présence : BCE stable sur logits, positifs et négatifs normalisés séparément.
/// Boîtes : Smooth L1 sur les seules cellules positives, après décodage.
pub fn detection_loss(predictions: Tensor<4>, batch: &DetectionBatch) -> Tensor<1> {
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
    let error = (Tensor::cat(vec![offsets, sizes], 1) - batch.boxes.clone()).abs();
    let quadratic = error.clone().clamp_max(1.0);
    let smooth_l1 = quadratic.clone() * quadratic.clone() * 0.5 + error - quadratic;
    let box_loss = (smooth_l1 * positive).sum() / (positive_count * 4.0);
    presence_loss + box_loss * 5.0
}
