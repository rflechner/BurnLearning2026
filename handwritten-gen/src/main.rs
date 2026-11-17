mod mnist;

use crate::mnist::{download_mnist_data, inspect_mnist_files, MnistDownloadResult};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match download_mnist_data()? {
        MnistDownloadResult::Success => {
            println!("Données MNIST téléchargées avec succès.");
        }
        MnistDownloadResult::Skip => println!("[Skip] - Données MNIST déjà téléchargées."),
    }

    inspect_mnist_files()?;

    Ok(())
}

