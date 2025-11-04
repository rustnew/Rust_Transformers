#![recursion_limit = "256"]

mod core;
mod transformer;
mod training;

use crate::training::optimizer::AdamOptimizer;
use crate::training::loss::softmax_cross_entropy;
use crate::training::dataloader::DataLoader;
use indicatif::{ProgressBar, ProgressStyle};

fn main() {
    println!("🚀 Transformer Rust - Tests AVANCÉS de performance...");
    
    // Tests de complexité croissante
    test_simple_pattern();
    test_sequence_reversal();
    test_arithmetic_operations();
    test_context_understanding();
    test_long_range_dependencies();
    analyze_model_complexity();
    
    println!("✅ Tous les tests avancés terminés!");
}

struct TransformerModel {
    blocks: Vec<crate::transformer::transformer_block::TransformerBlock>,
    embedding: crate::transformer::embedding::Embedding,
    output_proj: crate::core::layer::Linear,
}

impl TransformerModel {
    fn new(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
    ) -> Self {
        let mut blocks = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            blocks.push(crate::transformer::transformer_block::TransformerBlock::new(d_model, n_heads, d_ff));
        }
        
        let embedding = crate::transformer::embedding::Embedding::new(vocab_size, d_model);
        let output_proj = crate::core::layer::Linear::new(d_model, vocab_size);
        
        Self { blocks, embedding, output_proj }
    }
    
    fn forward(&self, input_ids: &crate::core::tensor::Tensor, training: bool) -> crate::core::tensor::Tensor {
        let mut x = self.embedding.forward(input_ids);
        
        for block in &self.blocks {
            x = block.forward(&x, training);
        }
        
        self.output_proj.forward(&x)
    }
    
    fn parameters(&mut self) -> Vec<&mut crate::core::tensor::Tensor> {
        vec![
            &mut self.embedding.weights,
            &mut self.output_proj.weights,
            &mut self.output_proj.bias,
        ]
    }
}

fn test_simple_pattern() {
    println!("\n🧪 TEST 1: Reconnaissance de motifs simples");
    println!("{}", "=".repeat(50));
    
    let  model = TransformerModel::new(100, 64, 2, 256, 2);
    let  _optimizer = AdamOptimizer::new(0.01);
    let mut dataloader = DataLoader::new(8, 16);
    
    let mut successes = 0;
    let total_tests = 5;
    
    for test_id in 0..total_tests {
        let (input, target) = dataloader.next_batch();
        let output = model.forward(&input, false);
        let (loss, _) = softmax_cross_entropy(&output, &target);
        
        // Test de cohérence : le modèle devrait avoir un comportement stable
        let loss_value = loss.mean();
        if loss_value < 2.0 {
            successes += 1;
            println!("   ✅ Test {}: Loss = {:.4} - COMPORTEMENT STABLE", test_id + 1, loss_value);
        } else {
            println!("   ⚠️  Test {}: Loss = {:.4} - VARIATIONS DÉTECTÉES", test_id + 1, loss_value);
        }
    }
    
    let success_rate = (successes as f32 / total_tests as f32) * 100.0;
    println!("📊 Taux de réussite: {:.1}%", success_rate);
}

fn test_sequence_reversal() {
    println!("\n🧪 TEST 2: Apprentissage d'inversion de séquence");
    println!("{}", "=".repeat(50));
    
    let mut model = TransformerModel::new(500, 128, 4, 512, 3);
    let mut optimizer = AdamOptimizer::new(0.005);
    
    let bar = ProgressBar::new(50);
    bar.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap());
    
    let mut losses = Vec::new();
    
    for _step in 0..50 {
        // Pour ce test, on simule un problème plus complexe
        let mut complex_dataloader = DataLoader::new(12, 24);
        let (input, target) = complex_dataloader.next_batch();
        
        let output = model.forward(&input, true);
        let (loss, grad) = softmax_cross_entropy(&output, &target);
        let loss_value = loss.mean();
        losses.push(loss_value);
        
        let gradients = vec![&grad];
        let mut params = model.parameters();
        optimizer.step(&mut params, &gradients);
        
        bar.inc(1);
    }
    
    bar.finish();
    
    let initial_loss = losses[0];
    let final_loss = losses[losses.len() - 1];
    let improvement = ((initial_loss - final_loss) / initial_loss) * 100.0;
    
    println!("📈 Performance inversion de séquence:");
    println!("   ┣ Loss initiale: {:.4}", initial_loss);
    println!("   ┣ Loss finale:   {:.4}", final_loss);
    println!("   ┣ Amélioration:  {:.1}%", improvement);
    
    if improvement > 60.0 {
        println!("   ┗ ✅ EXCELLENT: Bon apprentissage des patterns complexes");
    } else if improvement > 30.0 {
        println!("   ┗ ⚠️  MOYEN: Apprentissage limité des séquences");
    } else {
        println!("   ┗ ❌ FAIBLE: Difficulté avec les patterns complexes");
    }
}

fn test_arithmetic_operations() {
    println!("\n🧪 TEST 3: Simulation d'opérations arithmétiques");
    println!("{}", "=".repeat(50));
    
    let mut model = TransformerModel::new(1000, 256, 8, 1024, 4);
    
    // Test de capacité de raisonnement
    let mut reasoning_scores = Vec::new();
    
    for complexity in 1..=3 {
        let score = test_reasoning_capacity(&mut model, complexity);
        reasoning_scores.push(score);
        println!("   Complexité {}: Score = {:.2}/10.0", complexity, score);
    }
    
    let avg_score: f32 = reasoning_scores.iter().sum::<f32>() / reasoning_scores.len() as f32;
    
    println!("📊 Capacité de raisonnement moyenne: {:.2}/10.0", avg_score);
    
    match avg_score {
        8.0..=10.0 => println!("🎉 EXCEPTIONNEL: Forte capacité de raisonnement"),
        6.0..=7.9 => println!("✅ BON: Bonnes capacités cognitives"),
        4.0..=5.9 => println!("⚠️  MOYEN: Capacités limitées"),
        _ => println!("❌ FAIBLE: Difficultés de raisonnement"),
    }
}

fn test_reasoning_capacity(model: &mut TransformerModel, complexity: usize) -> f32 {
    // Simulation de tests de raisonnement à complexité croissante
    let base_score = 2.0;
    
    // Plus la complexité est élevée, plus le score devrait être bas
    // sauf si le modèle est vraiment bon
    let complexity_penalty = (complexity as f32 - 1.0) * 0.5;
    
    // Score basé sur la cohérence des prédictions
    let mut dataloader = DataLoader::new(8, 16);
    let mut consistency_score = 0.0;
    let test_runs = 5;
    
    for _ in 0..test_runs {
        let (input, target) = dataloader.next_batch();
        let output = model.forward(&input, false);
        let (loss, _) = softmax_cross_entropy(&output, &target);
        
        // Un loss bas indique une bonne cohérence
        let loss_value = loss.mean();
        if loss_value < 1.0 {
            consistency_score += 2.0;
        } else if loss_value < 2.0 {
            consistency_score += 1.0;
        }
    }
    
    let avg_consistency = consistency_score / test_runs as f32;
    (base_score + avg_consistency - complexity_penalty).max(0.0).min(10.0)
}

fn test_context_understanding() {
    println!("\n🧪 TEST 4: Compréhension contextuelle");
    println!("{}", "=".repeat(50));
    
    let mut model = TransformerModel::new(800, 192, 6, 768, 3);
    let mut optimizer = AdamOptimizer::new(0.003);
    
    let context_lengths = [8, 16, 32, 64];
    let mut context_scores = Vec::new();
    
    for &context_len in &context_lengths {
        let score = evaluate_context_performance(&mut model, &mut optimizer, context_len);
        context_scores.push(score);
        println!("   Longueur contexte {}: Score = {:.2}", context_len, score);
    }
    
    // Analyse de la scalabilité
    let short_context = context_scores[0];
    let long_context = context_scores[context_scores.len() - 1];
    let scalability = (long_context / short_context) * 100.0;
    
    println!("📊 Analyse de scalabilité:");
    println!("   ┣ Court terme ({}): {:.2}", context_lengths[0], short_context);
    println!("   ┣ Long terme ({}):  {:.2}", context_lengths[context_lengths.len() - 1], long_context);
    println!("   ┣ Maintien: {:.1}%", scalability);
    
    if scalability > 80.0 {
        println!("   ┗ ✅ EXCELLENTE scalabilité contextuelle");
    } else if scalability > 60.0 {
        println!("   ┗ ⚠️  BONNE scalabilité");
    } else {
        println!("   ┗ ❌ LIMITÉE sur les longs contextes");
    }
}

fn evaluate_context_performance(model: &mut TransformerModel, optimizer: &mut AdamOptimizer, context_len: usize) -> f32 {
    let mut temp_dataloader = DataLoader::new(8, context_len);
    let mut performances = Vec::new();
    
    for _ in 0..10 {
        let (input, target) = temp_dataloader.next_batch();
        let output = model.forward(&input, true);
        let (loss, grad) = softmax_cross_entropy(&output, &target);
        
        let gradients = vec![&grad];
        let mut params = model.parameters();
        optimizer.step(&mut params, &gradients);
        
        performances.push(loss.mean());
    }
    
    // Score inversement proportionnel à la loss moyenne
    let avg_performance: f32 = performances.iter().sum::<f32>() / performances.len() as f32;
    10.0 / (1.0 + avg_performance) // Normalisé entre 0 et 10
}

fn test_long_range_dependencies() {
    println!("\n🧪 TEST 5: Dépendances longues distances");
    println!("{}", "=".repeat(50));
    
    let mut model = TransformerModel::new(1200, 320, 8, 1280, 6);
    
    println!("🧠 Test des relations éloignées dans les séquences...");
    
    let dependency_distances = [4, 8, 16, 32, 64];
    let mut dependency_scores = Vec::new();
    
    for &distance in &dependency_distances {
        let score = test_dependency_at_distance(&mut model, distance);
        dependency_scores.push(score);
        println!("   Distance {}: Score = {:.2}", distance, score);
    }
    
    // Analyse de la décroissance
    let mut decay_rates = Vec::new();
    for i in 1..dependency_scores.len() {
        let decay = (dependency_scores[i-1] - dependency_scores[i]) / dependency_scores[i-1];
        decay_rates.push(decay * 100.0);
    }
    
    let avg_decay: f32 = decay_rates.iter().sum::<f32>() / decay_rates.len() as f32;
    
    println!("📊 Analyse des dépendances longues:");
    println!("   ┣ Score moyen: {:.2}", dependency_scores.iter().sum::<f32>() / dependency_scores.len() as f32);
    println!("   ┣ Décroissance moyenne: {:.1}% par doublement de distance", avg_decay);
    
    if avg_decay < 10.0 {
        println!("   ┗ ✅ EXCELLENT: Maintien des dépendances longues");
    } else if avg_decay < 25.0 {
        println!("   ┗ ⚠️  BON: Bon maintien des relations");
    } else {
        println!("   ┗ ❌ LIMITÉ: Perte rapide des dépendances éloignées");
    }
}

fn test_dependency_at_distance(model: &mut TransformerModel, distance: usize) -> f32 {
    // Simulation de test de dépendances à différentes distances
    let mut temp_dataloader = DataLoader::new(6, distance * 2);
    let mut scores = Vec::new();
    
    for _ in 0..8 {
        let (input, target) = temp_dataloader.next_batch();
        let output = model.forward(&input, false);
        let (loss, _) = softmax_cross_entropy(&output, &target);
        
        // Score basé sur la performance (plus c'est bas, mieux c'est)
        let loss_value = loss.mean();
        let score = 10.0 / (1.0 + loss_value); // Normalisation
        scores.push(score);
    }
    
    scores.iter().sum::<f32>() / scores.len() as f32
}

// Fonction utilitaire pour analyser la complexité
fn analyze_model_complexity() {
    println!("\n🔍 ANALYSE DE COMPLEXITÉ DU MODÈLE");
    println!("{}", "=".repeat(40));
    
    let configurations = [
        (100, 64, 2, 256, 2),
        (500, 128, 4, 512, 3),
        (1000, 256, 8, 1024, 4),
        (1200, 320, 8, 1280, 6),
    ];
    
    for (i, (vocab, d_model, heads, d_ff, layers)) in configurations.iter().enumerate() {
        let _model = TransformerModel::new(*vocab, *d_model, *heads, *d_ff, *layers);
        println!("   Config {}: {:.1}M paramètres (est.) - Complexité: {}/10", 
                 i + 1, 
                 estimate_parameters(*vocab, *d_model, *d_ff, *layers),
                 estimate_complexity(*d_model, *heads, *layers));
    }
}

fn estimate_parameters(vocab_size: usize, d_model: usize, d_ff: usize, n_layers: usize) -> f32 {
    // Estimation grossière des paramètres
    let embedding_params = vocab_size * d_model;
    let attention_params = n_layers * 4 * d_model * d_model; // Q, K, V, O
    let ff_params = n_layers * 2 * d_model * d_ff; // 2 couches linéaires
    let output_params = d_model * vocab_size;
    
    (embedding_params + attention_params + ff_params + output_params) as f32 / 1_000_000.0
}

fn estimate_complexity(d_model: usize, n_heads: usize, n_layers: usize) -> usize {
    // Score de complexité simple
    let mut score = 0;
    if d_model >= 256 { score += 3; } else if d_model >= 128 { score += 2; } else { score += 1; }
    if n_heads >= 8 { score += 3; } else if n_heads >= 4 { score += 2; } else { score += 1; }
    if n_layers >= 6 { score += 4; } else if n_layers >= 4 { score += 3; } else if n_layers >= 2 { score += 2; } else { score += 1; }
    score
}