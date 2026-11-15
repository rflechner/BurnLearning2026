mod api;
mod dataset;
mod device;

use burn::tensor::Tensor;
use device::default_device;

/// How many images to fully decode; the rest are only checked for existence.
const DECODE_LIMIT: usize = 50;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    let device = default_device();
    println!("Using device: {:?}", device);

    let tensor1 = Tensor::<2>::from_data([[2., 3.], [4., 5.]], &device);
    let tensor2 = Tensor::<2>::ones_like(&tensor1);

    // Print the element-wise addition of the two tensors.
    println!("{}", tensor1 + tensor2);

    let annotations_path = std::env::var_os("ANNOTATIONS_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(dataset::ANNOTATIONS_PATH)
        });
    let images_root_path = std::env::var_os("IMAGES_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(dataset::images_root())
        });

    if std::env::args().any(|arg| arg == "--inspect") {
        inspect_dataset(&*annotations_path, &*images_root_path);
        return Ok(());
    }
    let state = api::DatasetState::load(&annotations_path, &images_root_path)?;
    let address = std::env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:3001".into());
    let listener = tokio::net::TcpListener::bind(&address).await?;
    println!("Dataset API: http://{}", listener.local_addr()?);
    axum::serve(listener, api::router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

fn inspect_dataset(annotations_path: &std::path::Path, images_root : &std::path::Path) {
    let train_data =
        dataset::load_annotations(annotations_path).expect("open annotations");
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
    println!(
        "checked={checked} faces={faces} decoded={decoded} missing={missing} annotation_errors={errors}"
    );
}
