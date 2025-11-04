use super::tensor::Tensor;

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Gelu,
}

impl Activation {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        match self {
            Activation::Gelu => self.gelu(input),
        }
    }
    
    fn gelu(&self, input: &Tensor) -> Tensor {
        // Approximation GELU avec précision réduite
        let data = input.data.mapv(|x| {
            0.5 * x * (1.0 + (x * 0.797_884_6 * (1.0 + 0.044_715 * x * x)).tanh())
        });
        Tensor::new(data)
    }
}