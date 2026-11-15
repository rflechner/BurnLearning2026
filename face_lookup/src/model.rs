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
    activation: Relu,
}

impl Model {
    pub fn default(device: &burn::tensor::Device) -> Model {
        Model {
            conv1: Conv2dConfig::new([3, 16], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv2: Conv2dConfig::new([16, 32], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv3: Conv2dConfig::new([32, 64], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv4: Conv2dConfig::new([64, 64], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            head: Conv2dConfig::new([64, 5], [1, 1]).init(device),
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, images: burn::tensor::Tensor<4>) -> burn::tensor::Tensor<4> {
        let x = self.activation.forward(self.conv1.forward(images));
        let x = self.activation.forward(self.conv2.forward(x));
        let x = self.activation.forward(self.conv3.forward(x));
        let x = self.activation.forward(self.conv4.forward(x));

        self.head.forward(x)
    }
}
