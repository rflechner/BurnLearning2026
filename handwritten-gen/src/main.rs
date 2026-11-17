mod mnist;

use crate::mnist::{MnistDownloadResult, download_mnist_data};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match download_mnist_data()? {
        MnistDownloadResult::Success => {
            println!("Données MNIST téléchargées avec succès.");
        }
        MnistDownloadResult::Skip => println!("[Skip] - Données MNIST déjà téléchargées."),
    }
    
    Ok(())
}

