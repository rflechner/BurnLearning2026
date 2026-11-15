use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{PaddingConfig2d, Relu};

#[derive(Module, Debug)]
pub struct Model {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv4: Conv2d,
    head: Conv2d,
    activation: Relu
}

impl Model {
    pub fn default(device: &burn::tensor::Device) -> Model {
        Model {
            conv1: Conv2dConfig::new([3, 16], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv2: Conv2dConfig::new([3, 32], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv3: Conv2dConfig::new([3, 64], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv4: Conv2dConfig::new([3, 64], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            head: Conv2dConfig::new([64, 5], [1, 1]).init(device),
            activation: Relu::new(),
        }
    }
}