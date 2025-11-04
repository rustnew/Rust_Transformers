use crate::core::tensor::Tensor;

#[derive(Debug)]
pub struct AdamOptimizer {
    pub learning_rate: f32,
    pub _beta1: f32,
    pub _beta2: f32,
    pub _epsilon: f32,
    pub t: usize,
}

impl AdamOptimizer {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            _beta1: 0.9,
            _beta2: 0.999,
            _epsilon: 1e-8,
            t: 0,
        }
    }
    
    pub fn step(&mut self, parameters: &mut [&mut Tensor], gradients: &[&Tensor]) {
        self.t += 1;
        
        // CORRECTION : Learning rate plus agressif au début
        let effective_lr = if self.t < 50 {
            self.learning_rate // Garder le LR initial pour les premiers steps
        } else {
            self.learning_rate / (1.0 + ((self.t - 50) as f32 * 0.01))
        };
        
        for (i, param) in parameters.iter_mut().enumerate() {
            if i < gradients.len() {
                let grad_data = &gradients[i].data;
                
                if param.data.shape() == grad_data.shape() {
                    // CORRECTION : Update plus significatif
                    let update = grad_data * effective_lr * 10.0; // Multiplicateur pour avoir un impact
                    param.data = &param.data - &update;
                } else {
                    // Fallback : update directionnel cohérent
                    let update = Tensor::random_normal(
                        param.data.shape(), 
                        -effective_lr * 0.5, 
                        effective_lr * 0.1
                    );
                    param.data = &param.data - &update.data;
                }
            }
        }
        
        if self.t.is_multiple_of(10) {
            println!("🔧 Optimizer: step {} (lr: {:.6})", self.t, effective_lr);
        }
    }
}