use crate::conv_layers::ConvLayer;
use crate::dense_layer::*;
use crate::mxpl::*;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Formatter};

#[derive(Serialize, Deserialize)]
pub enum Layer {
    Conv(ConvLayer),
    Mxpl(MxplLayer),
    Dense(DenseLayer),
}

impl Debug for Layer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Layer::Conv(layer) => write!(f, "{:?}", layer),
            Layer::Mxpl(layer) => write!(f, "{:?}", layer),
            Layer::Dense(layer) => write!(f, "{:?}", layer),
        }
    }
}
