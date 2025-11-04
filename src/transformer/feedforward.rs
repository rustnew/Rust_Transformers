use crate::core::tensor::Tensor;
use crate::core::layer::Linear;
use crate::core::activation::Activation;

#[derive(Debug)]
pub struct FeedForward {
    pub linear1: Linear,
    pub linear2: Linear,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let linear1 = Linear::new(d_model, d_ff).with_activation(Activation::Gelu);
        let linear2 = Linear::new(d_ff, d_model);
        
        Self {
            linear1,
            linear2,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let hidden = self.linear1.forward(x);
        self.linear2.forward(&hidden)
    }
}