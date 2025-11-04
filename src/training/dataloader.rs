use crate::core::tensor::Tensor;
use ndarray::Array;
use rand::Rng;

#[derive(Debug)]
pub struct DataLoader {
    pub batch_size: usize,
    pub seq_len: usize,
    pub current_step: usize,
    pub vocab_size: usize,
}

impl DataLoader {
    pub fn new(batch_size: usize, seq_len: usize) -> Self {
        Self {
            batch_size,
            seq_len,
            current_step: 0,
            vocab_size: 1000,  // Réduit pour correspondre au main
        }
    }
    
    pub fn next_batch(&mut self) -> (Tensor, Tensor) {
        // Génère des données CORRÉLÉES pour l'apprentissage
        let mut input_data = Array::zeros((self.batch_size, self.seq_len));
        let mut target_data = Array::zeros((self.batch_size, self.seq_len));
        
        let mut rng = rand::rng();
        
        for i in 0..self.batch_size {
            for j in 0..self.seq_len {
                // Crée une relation simple: target = input + 1 (mod vocab_size)
                let input_val = rng.random_range(0..self.vocab_size - 1);
                let target_val = (input_val + 1) % self.vocab_size;
                
                input_data[[i, j]] = input_val as f32;
                target_data[[i, j]] = target_val as f32;
            }
        }
        
        self.current_step += 1;
        (
            Tensor::new(input_data.into_dyn()),
            Tensor::new(target_data.into_dyn())
        )
    }
}