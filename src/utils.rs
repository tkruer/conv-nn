use image::ImageReader;
use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Represents a training image, which can be stored as a loaded image or a file path.
#[derive(Serialize, Deserialize, Clone)]
pub enum TrainImage {
    Image(Array3<f32>),
    Path(PathBuf),
}

/// Holds training and testing data along with metadata.
#[derive(Serialize, Deserialize, Default)]
pub struct TrainingData {
    pub trn_img: Vec<TrainImage>,
    pub trn_lbl: Vec<usize>,
    pub tst_img: Vec<TrainImage>,
    pub tst_lbl: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
    pub trn_size: usize,
    pub tst_size: usize,
    pub classes: HashMap<usize, usize>,
}

/// Computes the outer product of two vectors.
pub fn outer(x: Array1<f32>, y: Array1<f32>) -> Array2<f32> {
    Array2::from_shape_fn((x.len(), y.len()), |(i, j)| x[i] * y[j])
}

/// Provides mappings between state names and their indices.
pub struct StateMapper;

/// Defines when the model should be saved.
#[derive(Serialize, Deserialize)]
pub enum SavingStrategy {
    EveryEpoch(bool),
    EveryNthEpoch(bool, f32),
    BestTrainingAccuracy(bool),
    BestTestingAccuracy(bool),
    Never,
}

/// Loads an image from a specified path and converts it to an `Array3<f32>`.
pub fn load_image(path: &Path) -> Result<Array3<f32>, String> {
    let img = ImageReader::open(path)
        .map_err(|e| e.to_string())?
        .decode()
        .map_err(|e| e.to_string())?;
    let img = img.to_rgb8();

    let (rows, cols) = (img.height() as usize, img.width() as usize);
    let mut array = Array3::<f32>::zeros((rows, cols, 3));

    for (x, y, pixel) in img.enumerate_pixels() {
        let (r, g, b) = (pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
        array[[x as usize, y as usize, 0]] = r / 255.0;
        array[[x as usize, y as usize, 1]] = g / 255.0;
        array[[x as usize, y as usize, 2]] = b / 255.0;
    }

    Ok(array)
}
