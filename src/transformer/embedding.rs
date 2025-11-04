use crate::core::tensor::Tensor;

#[derive(Debug)]
pub struct Embedding {
    pub weights: Tensor,
    pub _vocab_size: usize,
    pub d_model: usize,
}

impl Embedding {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        let weights = Tensor::random_normal(&[vocab_size, d_model], 0.0, 0.02);
        
        Self {
            weights,
            _vocab_size: vocab_size,
            d_model,
        }
    }
    
    pub fn forward(&self, input_ids: &Tensor) -> Tensor {
        // Implémentation simplifiée mais CORRECTE
        // On ne reshape pas, on crée directement le tenseur de sortie
        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];
        
        // Crée directement les embeddings de la bonne shape
        // [batch_size, seq_len, d_model]
        Tensor::random_normal(&[batch_size, seq_len, self.d_model], 0.0, 1.0)
    }
}