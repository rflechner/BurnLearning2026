mod device;
mod dataset;

use burn::tensor::Tensor;
use device::default_device;

fn main() {
    let device = default_device();
    println!("Using device: {:?}", device);

    let tensor1 = Tensor::<2>::from_data([[2., 3.], [4., 5.]], &device);
    let tensor2 = Tensor::<2>::ones_like(&tensor1);

    // Print the element-wise addition of the two tensors.
    println!("{}", tensor1 + tensor2);
    
    let train_data = dataset::load_annotations("data/extracted/db171f1b7fedf4d3453e81297ff02f9915356d19/wider_face_split/wider_face_train_bbx_gt.txt");
    if let Ok(train_data) = train_data {
        for image in train_data {
            if let Ok(mut image) = image {
                println!("image file: {:?}", image.image_file_path);
            }
        }
    }
}
