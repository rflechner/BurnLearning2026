use burn::tensor::{Device, Tensor};

/// Best available GPU device per platform: CUDA on Windows, Metal on macOS, wgpu elsewhere.
fn default_device() -> Device {
    #[cfg(target_os = "windows")]
    {
        Device::cuda(0)
    }
    #[cfg(target_os = "macos")]
    {
        use burn::tensor::DeviceKind;
        Device::metal(DeviceKind::IntegratedGpu(0))
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        Device::wgpu(Default::default())
    }
}

fn main() {
    let device = default_device();
    println!("Using device: {:?}", device);

    let tensor1 = Tensor::<2>::from_data([[2., 3.], [4., 5.]], &device);
    let tensor2 = Tensor::<2>::ones_like(&tensor1);

    // Print the element-wise addition of the two tensors.
    println!("{}", tensor1 + tensor2);
}