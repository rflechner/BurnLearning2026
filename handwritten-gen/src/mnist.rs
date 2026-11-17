use std::fs::{File, DirBuilder, DirEntry, ReadDir};
use std::io::Write;
use std::path::Path;

pub enum MnistDownloadResult {
    Success,
    Skip,
}
static parquet_test: &str = "test-00000-of-00001.parquet";
static parquet_train: &str = "train-00000-of-00001.parquet";

pub fn download_mnist_data() -> Result<MnistDownloadResult, Box<dyn std::error::Error>> {
    let current_dir = std::env::current_dir()?;
    let mut mnist_dir = current_dir.clone();
    mnist_dir.push("mnist");
    let parquet_train_full_path = mnist_dir.join(parquet_train);
    let parquet_test_full_path = mnist_dir.join(parquet_test);

    if parquet_train_full_path.exists() && parquet_test_full_path.exists() {
        return Ok(MnistDownloadResult::Skip);
    }

    DirBuilder::new().recursive(true).create(mnist_dir)?;

    // let client = reqwest::blocking::Client::new();
    //
    // let mut test_response = client
    //     .get("https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet?download=true")
    //     .send()?
    //     .error_for_status()?;
    //
    // let test_binary_content = &test_response.bytes()?;
    //
    // let mut test_file = File::create(parquet_test_full_path)?;
    // test_file.write_all(test_binary_content)?;

    download_mnist_file(parquet_test_full_path.as_path(), "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet?download=true")?;

    download_mnist_file(parquet_train_full_path.as_path(), "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/train-00000-of-00001.parquet?download=true")?;

    // let mut train_response = client
    //     .get("https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/train-00000-of-00001.parquet?download=true")
    //     .send()?
    //     .error_for_status()?;
    //
    // let train_binary_content = &train_response.bytes()?;
    //
    // let mut train_file = File::create(parquet_train_full_path)?;
    // train_file.write_all(train_binary_content)?;

    Ok(MnistDownloadResult::Success)
}

fn download_mnist_file (full_destination_path: &Path, url: &str) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::new();
    let response = client
        .get(url)
        .send()?
        .error_for_status()?;

    let binary_content = &response.bytes()?;

    let mut file = File::create(full_destination_path)?;
    file.write_all(binary_content)?;

    Ok(())
}
