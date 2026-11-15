mod device;
mod dataset;

use std::path::{Path, PathBuf};

use burn::tensor::Tensor;
use device::default_device;


/// How many images to fully decode; the rest are only checked for existence.
const DECODE_LIMIT: usize = 50;

fn main() {
    let device = default_device();
    println!("Using device: {:?}", device);

    let tensor1 = Tensor::<2>::from_data([[2., 3.], [4., 5.]], &device);
    let tensor2 = Tensor::<2>::ones_like(&tensor1);

    // Print the element-wise addition of the two tensors.
    println!("{}", tensor1 + tensor2);

    let train_data = dataset::load_annotations(dataset::ANNOTATIONS_PATH).expect("open annotations");
    let images_root = dataset::images_root();
    let mut checked = 0;
    let mut faces = 0;
    let mut missing = 0;
    let mut decoded = 0;
    let mut errors = 0;
    for entry in train_data {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                errors += 1;
                eprintln!("annotation error: {e}");
                continue;
            }
        };
        checked += 1;
        faces += entry.faces.len();

        let path = images_root.join(&entry.image_file_path);
        if !path.exists() {
            missing += 1;
            eprintln!("missing image: {}", entry.image_file_path);
            continue;
        }
        if decoded < DECODE_LIMIT {
            match image::open(&path) {
                Ok(img) => {
                    println!(
                        "decoded {} -> {}x{} px, {} face(s)",
                        entry.image_file_path,
                        img.width(),
                        img.height(),
                        entry.faces.len()
                    );
                    decoded += 1;
                }
                Err(e) => eprintln!("failed to decode {}: {e}", entry.image_file_path),
            }
        }
    }
    println!("checked={checked} faces={faces} decoded={decoded} missing={missing} annotation_errors={errors}");
}
