use conv_nn::activation::Activation;
use conv_nn::cnn::*;
use conv_nn::mnist_impl::*;
use conv_nn::optimizer::*;

fn main() {
    // Load MNIST dataset
    let data = load_mnist("./data/");

    // Set hyperparameters
    let hyperparameters = Hyperparameters {
        batch_size: 10,
        epochs: 10,
        optimizer: OptimizerAlg::SGD(0.1),
        ..Hyperparameters::default()
    };

    // Create CNN architecture
    let mut cnn = CNN::new(data, hyperparameters);
    cnn.set_input_shape(vec![28, 28, 3]);
    cnn.add_conv_layer(8, 3);
    cnn.add_mxpl_layer(2);
    cnn.add_dense_layer(128, Activation::Relu, Some(0.25));
    cnn.add_dense_layer(64, Activation::Relu, Some(0.25));
    cnn.add_dense_layer(10, Activation::Softmax, None);

    cnn.train();
}
