#[cfg(test)]
mod tests {
    use conv_nn::activation::*;
    use ndarray::array;

    #[test]
    fn test_relu_forward() {
        let input = array![-1.0, 0.0, 1.0, 2.0];
        let expected = array![0.0, 0.0, 1.0, 2.0];
        let output = forward(input.clone(), Activation::Relu);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_relu_backward() {
        let input = array![-1.0, 0.0, 1.0, 2.0];
        let expected = array![0.0, 0.0, 1.0, 1.0];
        let output = backward(input.clone(), Activation::Relu);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_sigmoid_forward() {
        let input = array![0.0, 1.0, -1.0];
        let expected = array![
            0.5,                           // sigmoid(0)
            1.0 / (1.0 + (-1.0f32).exp()), // sigmoid(1)
            1.0 / (1.0 + 1.0f32.exp())     // sigmoid(-1)
        ];
        let output = forward(input.clone(), Activation::Sigmoid);
        assert!((output - expected).mapv(f32::abs).sum() < 1e-6);
    }

    #[test]
    fn test_sigmoid_backward() {
        let input = array![0.5, 0.7, 0.2];
        let expected = input.mapv(|x| x * (1.0 - x));
        let output = backward(input.clone(), Activation::Sigmoid);
        assert!((output - expected).mapv(f32::abs).sum() < 1e-6);
    }

    #[test]
    fn test_softmax_forward() {
        let input = array![1.0, 2.0, 3.0];
        let output = forward(input.clone(), Activation::Softmax);
        let sum: f32 = output.sum();
        assert!((sum - 1.0).abs() < 1e-6, "Softmax output does not sum to 1");
    }

    #[test]
    fn test_softmax_derivative() {
        let input = array![1.0, 2.0, 3.0];
        let output = backward(input.clone(), Activation::Softmax);
        let expected = array![1.0, 1.0, 1.0]; // Currently softmax_derivative just returns ones
        assert_eq!(output, expected);
    }
}
