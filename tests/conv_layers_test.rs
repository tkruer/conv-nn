#[cfg(test)]
mod tests {    
    use conv_nn::conv_layers::ConvLayer;
    use conv_nn::optimizer::OptimizerAlg;
    use ndarray::Array3;

    #[test]
    fn test_conv_layer_initialization() {
        let input_size = (28, 28, 3);
        let kernel_size = 3;
        let stride = 1;
        let num_filters = 5;
        let optimizer_alg = OptimizerAlg::Adam(0.9, 0.999, 1e-8);

        let conv_layer = ConvLayer::new(input_size, kernel_size, stride, num_filters, optimizer_alg);

        assert_eq!(conv_layer.input_size, input_size);
        assert_eq!(conv_layer.kernel_size, kernel_size);
        assert_eq!(conv_layer.stride, stride);
        assert_eq!(conv_layer.num_filters, num_filters);
        assert_eq!(conv_layer.output_size, ((input_size.0 - kernel_size) / stride + 1, (input_size.1 - kernel_size) / stride + 1, num_filters));
    }

    #[test]
    fn test_conv_layer_forward_propagation() {
        let input_size = (5, 5, 3);
        let kernel_size = 3;
        let stride = 1;
        let num_filters = 2;
        let optimizer_alg = OptimizerAlg::Adam(0.9, 0.999, 1e-8);

        let mut conv_layer = ConvLayer::new(input_size, kernel_size, stride, num_filters, optimizer_alg);
        let input = Array3::<f32>::ones(input_size);

        let output = conv_layer.forward_propagate(input.clone());

        assert_eq!(output.shape(), &[3, 3, num_filters]); // Output size calculation
    }

    #[test]
    fn test_conv_layer_backward_propagation() {
        let input_size = (5, 5, 3);
        let kernel_size = 3;
        let stride = 1;
        let num_filters = 2;
        let optimizer_alg = OptimizerAlg::Adam(0.9, 0.999, 1e-8);

        let mut conv_layer = ConvLayer::new(input_size, kernel_size, stride, num_filters, optimizer_alg);
        let input = Array3::<f32>::ones(input_size);
        conv_layer.forward_propagate(input);

        let error = Array3::<f32>::ones(conv_layer.output_size);
        let backprop_error = conv_layer.back_propagate(error);

        assert_eq!(backprop_error.shape(), &[5, 5, 3]); // Should match input size
    }

    #[test]
    fn test_conv_layer_update() {
        let input_size = (5, 5, 3);
        let kernel_size = 3;
        let stride = 1;
        let num_filters = 2;
        let optimizer_alg = OptimizerAlg::Adam(0.9, 0.999, 1e-8);

        let mut conv_layer = ConvLayer::new(input_size, kernel_size, stride, num_filters, optimizer_alg);
        let input = Array3::<f32>::ones(input_size);
        conv_layer.forward_propagate(input);

        let error = Array3::<f32>::ones(conv_layer.output_size);
        conv_layer.back_propagate(error);

        let before_update = conv_layer.kernels.clone();
        conv_layer.update(1);

        assert_ne!(before_update, conv_layer.kernels); // Weights should change
    }

    #[test]
    fn test_conv_layer_zero_reset() {
        let input_size = (5, 5, 3);
        let kernel_size = 3;
        let stride = 1;
        let num_filters = 2;
        let optimizer_alg = OptimizerAlg::Adam(0.9, 0.999, 1e-8);

        let mut conv_layer = ConvLayer::new(input_size, kernel_size, stride, num_filters, optimizer_alg);
        conv_layer.zero();

        assert!(conv_layer.kernel_changes.iter().all(|&x| x == 0.0));
        assert!(conv_layer.output.iter().all(|&x| x == 0.0));
    }
}