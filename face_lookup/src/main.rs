use burn::tensor::{Device, Tensor};

fn main() {
    // Run on any device: Device::{wgpu, cuda, rocm, metal, vulkan, flex, cpu}
    let device = Device::wgpu(Default::default());

    let tensor1 = Tensor::<2>::from_data([[2., 3.], [4., 5.]], &device);
    let tensor2 = Tensor::<2>::ones_like(&tensor1);

    // Print the element-wise addition of the two tensors.
    println!("{}", tensor1 + tensor2);
}