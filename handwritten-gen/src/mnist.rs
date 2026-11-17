use std::fs::{File, DirBuilder, DirEntry, ReadDir};
use std::io::Write;
use std::path::{Path, PathBuf};
use parquet::file::reader::{FileReader, SerializedFileReader};

pub enum MnistDownloadResult {
    Success,
    Skip,
}
static parquet_test: &str = "test-00000-of-00001.parquet";
static parquet_train: &str = "train-00000-of-00001.parquet";

struct MnistFiles {
    mnist_dir: PathBuf,
    parquet_train_full_path: PathBuf,
    parquet_test_full_path: PathBuf,
}

fn get_mnist_files_path() -> Result<MnistFiles, Box<dyn std::error::Error>> {
    let current_dir = std::env::current_dir()?;
    let mut mnist_dir = current_dir.clone();
    mnist_dir.push("mnist");
    let parquet_train_full_path = mnist_dir.join(parquet_train);
    let parquet_test_full_path = mnist_dir.join(parquet_test);

    Ok(MnistFiles {
        mnist_dir,
        parquet_train_full_path,
        parquet_test_full_path,
    })
}

pub fn download_mnist_data() -> Result<MnistDownloadResult, Box<dyn std::error::Error>> {
    let mnist_files = get_mnist_files_path()?;
    if mnist_files.parquet_train_full_path.exists() && mnist_files.parquet_test_full_path.exists() {
        return Ok(MnistDownloadResult::Skip);
    }

    DirBuilder::new().recursive(true).create(mnist_files.mnist_dir)?;

    download_mnist_file(mnist_files.parquet_test_full_path.as_path(), "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet?download=true")?;

    download_mnist_file(mnist_files.parquet_train_full_path.as_path(), "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/train-00000-of-00001.parquet?download=true")?;

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

pub fn inspect_mnist_files() -> Result<(), Box<dyn std::error::Error>> {
    let mnist_files = get_mnist_files_path()?;
    inspect_mnist(&mnist_files.parquet_train_full_path)?;
    inspect_mnist(&mnist_files.parquet_test_full_path)?;

    Ok(())
}

pub fn inspect_mnist(full_file_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(full_file_path)?;
    let reader = SerializedFileReader::new(file)?;

    let metadata = reader.metadata().file_metadata();

    println!("Nombre de lignes : {}", metadata.num_rows());
    println!("Schéma : {:#?}", metadata.schema());

    // None signifie : lire toutes les colonnes.
    let rows = reader.get_row_iter(None)?;

    // On se limite aux trois premiers exemples.
    for (index, row_result) in rows.take(3).enumerate() {
        let row = row_result?;
        println!("Exemple {index} : {row}");
    }

    Ok(())
}