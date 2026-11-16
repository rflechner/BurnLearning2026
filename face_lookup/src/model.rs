use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Initializer, PaddingConfig2d, Relu};
use burn::tensor::{Tensor, TensorData};

/// Score initial de présence (≈ 0,1) : sans cet a priori, les milliers de cellules de fond
/// démarrent à 0,5 et écrasent la perte des quelques positifs.
const PRESENCE_PRIOR: f32 = 0.1;
/// Taille initiale des boîtes en pixels de l’entrée du réseau.
const SIZE_PRIOR: f32 = 40.0;

/// Détecteur entièrement convolutionnel : pas de 8 px, 5 valeurs par cellule.
/// La même architecture accepte 320×320 (grille 40×40) ou 640×640 (grille 80×80).
/// La résolution du prétraitement est conservée séparément des paramètres appris.
///
/// Les trois convolutions à pas 2 descendent au pas 8 ; les convolutions dilatées qui suivent
/// élargissent le champ récepteur à 255 px sans réduire la grille. Avant, il plafonnait à
/// 31 px alors que la plupart des visages sont plus grands : une cellule ne voyait qu'un
/// morceau de peau et devinait la taille.
#[derive(Module, Debug)]
pub struct Model {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    context: Vec<Conv2d>,
    head: Conv2d,
    activation: Relu,
}

/// Initialisation de He, adaptée aux ReLU : la variance du signal reste stable avec la profondeur.
fn conv(
    device: &burn::tensor::Device,
    channels: [usize; 2],
    stride: usize,
    dilation: usize,
) -> Conv2d {
    Conv2dConfig::new(channels, [3, 3])
        .with_stride([stride, stride])
        .with_dilation([dilation, dilation])
        .with_padding(PaddingConfig2d::Explicit(
            dilation, dilation, dilation, dilation,
        ))
        .with_initializer(Initializer::KaimingNormal {
            gain: 2f64.sqrt(),
            fan_out_only: false,
        })
        .init(device)
}

impl Model {
    pub fn default(device: &burn::tensor::Device) -> Model {
        let mut head = Conv2dConfig::new([64, 5], [1, 1])
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.01,
            })
            .init(device);
        // Logit de présence, décalages (sigmoid(0) = centre de la cellule), log des tailles.
        let size = (SIZE_PRIOR / crate::batch::STRIDE).ln();
        let presence = (PRESENCE_PRIOR / (1.0 - PRESENCE_PRIOR)).ln();
        head.bias = Some(Param::from_tensor(Tensor::from_data(
            TensorData::new(vec![presence, 0.0, 0.0, size, size], [5]),
            device,
        )));
        Model {
            conv1: conv(device, [3, 16], 2, 1),
            conv2: conv(device, [16, 32], 2, 1),
            conv3: conv(device, [32, 64], 2, 1),
            // Champ récepteur : 15 px après conv3, puis +16, +32, +64, +128 px = 255 px.
            // Les dilatations 1, 2, 4, 8 se recouvrent sans laisser de trous dans la grille.
            context: [1, 2, 4, 8]
                .into_iter()
                .map(|dilation| conv(device, [64, 64], 1, dilation))
                .collect(),
            head,
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, images: Tensor<4>) -> Tensor<4> {
        let mut x = self.activation.forward(self.conv1.forward(images));
        x = self.activation.forward(self.conv2.forward(x));
        x = self.activation.forward(self.conv3.forward(x));
        for layer in &self.context {
            x = self.activation.forward(layer.forward(x));
        }
        self.head.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_grid_and_priors() {
        let device = crate::device::default_device();
        let model = Model::default(&device);
        let output = model.forward(Tensor::zeros([1, 3, 320, 320], &device));
        assert_eq!(output.dims(), [1, 5, 40, 40]);
        let values = output.into_data().try_to_vec::<f32>().unwrap();
        // Entrée nulle : la tête initialisée à 0,01 laisse presque uniquement les biais.
        let score = 1.0 / (1.0 + (-values[0]).exp());
        assert!(
            (score - PRESENCE_PRIOR).abs() < 0.02,
            "score initial {score}"
        );
        let width = values[3 * 1600].exp() * crate::batch::STRIDE;
        assert!((width - SIZE_PRIOR).abs() < 2.0, "largeur initiale {width}");
    }
}
