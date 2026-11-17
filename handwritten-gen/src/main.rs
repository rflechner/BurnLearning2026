mod mnist;

use std::fs::{File, DirBuilder, DirEntry, ReadDir};
use std::io::Write;
use image::{GrayImage, Luma, Rgb, RgbImage};
use crate::mnist::{download_mnist_data, MnistDownloadResult};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Une nouvelle image en niveaux de gris, initialement noire.
    let mut img = RgbImage::new(28, 28);

    // x = colonne, y = ligne. On trace une barre verticale blanche.
    for y in 5..23 {
        for x in 12..16 {
            //img.put_pixel(x, y, Luma([255u8]));
            img.put_pixel(x, y, Rgb([200u8, 100u8, 50u8]));
        }
    }

    img.save("premier-chiffre.png")?;
    println!("Image créée : premier-chiffre.png");

    match download_mnist_data()? {
        MnistDownloadResult::Success => {
            println!("Données MNIST téléchargées avec succès.");
        }
        MnistDownloadResult::Skip => println!("[Skip] - Données MNIST déjà téléchargées."),
    }
    
    
    Ok(())
}

