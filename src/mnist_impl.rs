use crate::utils::{load_image, TrainImage, TrainingData};
use ndarray::Array3;
use rand::seq::IteratorRandom;
use rust_mnist::Mnist;
use std::collections::HashMap;
use std::path::Path;

/// Loads the MNIST dataset and returns a structured `TrainingData` object.
pub fn load_mnist<T>(mnist_path: T) -> TrainingData
where
    T: AsRef<Path>,
{
    let (rows, cols) = (28, 28);
    let mnist_path = mnist_path.as_ref();
    let mnist = Mnist::new(mnist_path.to_str().unwrap());

    // Ensure "unpacked" directory exists
    let unpacked_path = mnist_path.join("unpacked");
    if !unpacked_path.exists() {
        std::fs::create_dir(&unpacked_path).expect("Failed to create unpacked folder.");
    }

    // Helper function to process images
    fn process_images(
        data: &Vec<[u8; 784]>,
        labels: &Vec<u8>,
        rows: usize,
        cols: usize,
    ) -> (Vec<TrainImage>, Vec<usize>) {
        let images = data
            .iter()
            .map(|img_data| {
                let mut img = Array3::<f32>::zeros((rows, cols, 1));
                for j in 0..rows {
                    for k in 0..cols {
                        img[[j, k, 0]] = img_data[(j * cols + k) as usize] as f32 / 255.0;
                    }
                }
                TrainImage::Image(img)
            })
            .collect();

        let labels = labels.iter().map(|&label| label as usize).collect();
        (images, labels)
    }

    // Process training and testing data
    let (trn_img, trn_lbl) = process_images(&mnist.train_data, &mnist.train_labels, rows, cols);
    let (tst_img, tst_lbl) = process_images(&mnist.test_data, &mnist.test_labels, rows, cols);

    // Classes map
    let classes: HashMap<usize, usize> = (0..10).enumerate().collect();

    TrainingData {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        rows,
        cols,
        trn_size: 60000,
        tst_size: 10000,
        classes,
    }
}

/// Retrieves a random training image and its label from the `TrainingData`.
pub fn get_random_image(data: &TrainingData) -> (Array3<f32>, usize) {
    get_random_sample(&data.trn_img, &data.trn_lbl)
}

/// Retrieves a random test image and its label from the `TrainingData`.
pub fn get_random_test_image(data: &TrainingData) -> (Array3<f32>, usize) {
    get_random_sample(&data.tst_img, &data.tst_lbl)
}

/// Helper function to fetch a random image and label from the provided dataset.
fn get_random_sample(images: &[TrainImage], labels: &[usize]) -> (Array3<f32>, usize) {
    let mut rng = rand::thread_rng();
    let (img, label) = images.iter().zip(labels.iter()).choose(&mut rng).unwrap();
    match img {
        TrainImage::Image(img) => (img.clone(), *label),
        TrainImage::Path(img_path) => {
            let img = load_image(img_path).expect("Failed to load image from path");
            (img, *label)
        }
    }
}
