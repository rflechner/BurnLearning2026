use burn::tensor::Device;

/// Best available GPU device per platform: CUDA on Windows, Metal on macOS, wgpu elsewhere.
pub fn default_device() -> Device {
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
