#[cfg(test)]
mod tests {
    use conv_nn::utils::{outer, SavingStrategy, TrainImage, TrainingData};
    use ndarray::{array, Array3};
    
    
    use std::path::PathBuf;

    #[test]
    fn test_training_data_initialization() {
        let training_data = TrainingData::default();

        assert_eq!(training_data.trn_img.len(), 0);
        assert_eq!(training_data.trn_lbl.len(), 0);
        assert_eq!(training_data.tst_img.len(), 0);
        assert_eq!(training_data.tst_lbl.len(), 0);
        assert_eq!(training_data.rows, 0);
        assert_eq!(training_data.cols, 0);
        assert_eq!(training_data.trn_size, 0);
        assert_eq!(training_data.tst_size, 0);
        assert_eq!(training_data.classes.len(), 0);
    }

    #[test]
    fn test_train_image_enum() {
        let path = PathBuf::from("test_image.png");
        let image_data = Array3::<f32>::zeros((28, 28, 3));

        let train_img_path = TrainImage::Path(path.clone());
        let train_img_data = TrainImage::Image(image_data.clone());

        match train_img_path {
            TrainImage::Path(p) => assert_eq!(p, path),
            _ => panic!("Expected TrainImage::Path"),
        }

        match train_img_data {
            TrainImage::Image(img) => assert_eq!(img.shape(), &[28, 28, 3]),
            _ => panic!("Expected TrainImage::Image"),
        }
    }

    #[test]
    fn test_outer_product() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![4.0, 5.0];

        let result = outer(x, y);

        let expected = array![[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_saving_strategy_enum() {
        let strategy1 = SavingStrategy::EveryEpoch(true);
        let strategy2 = SavingStrategy::EveryNthEpoch(false, 0.5);
        let strategy3 = SavingStrategy::BestTrainingAccuracy(true);
        let strategy4 = SavingStrategy::BestTestingAccuracy(false);
        let strategy5 = SavingStrategy::Never;

        match strategy1 {
            SavingStrategy::EveryEpoch(full_save) => assert!(full_save),
            _ => panic!("Expected EveryEpoch"),
        }

        match strategy2 {
            SavingStrategy::EveryNthEpoch(full_save, n) => {
                assert!(!full_save);
                assert_eq!(n, 0.5);
            }
            _ => panic!("Expected EveryNthEpoch"),
        }

        match strategy3 {
            SavingStrategy::BestTrainingAccuracy(full_save) => assert!(full_save),
            _ => panic!("Expected BestTrainingAccuracy"),
        }

        match strategy4 {
            SavingStrategy::BestTestingAccuracy(full_save) => assert!(!full_save),
            _ => panic!("Expected BestTestingAccuracy"),
        }

        match strategy5 {
            SavingStrategy::Never => {}
            _ => panic!("Expected Never"),
        }
    }
}
