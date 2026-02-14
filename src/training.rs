use anyhow::Result;
use tch::{nn, Device, Tensor, Kind};
use nn::OptimizerConfig;
use crate::model::Transformer;

/// Creates a causal mask for decoder self-attention.
/// This mask prevents positions from attending to future positions.
///
/// Args:
///   seq_len: The sequence length.
///   device: The device to create the mask on.
///
/// Returns:
///   A tensor of shape [1, 1, seq_len, seq_len] where future positions are masked.
fn create_causal_mask(seq_len: i64, device: Device) -> Tensor {
    // Create a lower triangular matrix: shape [seq_len, seq_len]
    // tril(0) keeps the diagonal and below (lower triangle)
    let ones = Tensor::ones(&[seq_len, seq_len], (Kind::Float, device));
    let mask = ones.tril(0);

    // Convert: 1 -> 0.0 (can attend), 0 -> -inf (cannot attend)
    // We want positions where mask is 0 to become -inf
    let large_negative = -1e9_f64;
    let mask = (&mask - 1.0) * large_negative;

    // Expand dimensions to [1, 1, seq_len, seq_len] for broadcasting
    // This will be broadcasted over batch_size and num_heads
    mask.unsqueeze(0).unsqueeze(0)
}

/// Implements a basic training loop for the Transformer model.
pub fn train<'a>(
    vs: &mut nn::VarStore,
    vocab_size: i64,
    d_model: i64,
    num_layers: i64,
    num_heads: i64,
    d_ff: i64,
    max_len: i64,
    batch_size: i64,
    seq_len: i64,
    num_epochs: i64,
    learning_rate: f64,
    batch_generator: &dyn Fn(i64, i64, Device) -> Result<(Tensor, Tensor)>,
    model_save_path: &str,
) -> Result<()> {
    let device = vs.device();

    // Initialize the Transformer model.
    let model = Transformer::new(
        &vs.root(), vocab_size, d_model, num_layers, num_heads, d_ff, max_len,
    );

    // Initialize the Adam optimizer.
    let mut opt = nn::Adam::default().build(vs, learning_rate)?;

    println!("Starting training for {} epochs...", num_epochs);

    // Create the causal mask once (it's the same for all batches with same seq_len)
    let causal_mask = create_causal_mask(seq_len, device);

    for epoch in 1..=num_epochs {
        // Generate a batch of data using the provided closure.
        let (inputs, targets) = batch_generator(batch_size, seq_len, device)?;

        // Forward pass: Get model predictions (logits) with causal mask
        let logits = model.forward(&inputs, Some(&causal_mask));

        // Calculate the loss.
        let loss = logits
            .view([-1, vocab_size])
            .cross_entropy_for_logits(&targets.view([-1]));

        // Zero gradients.
        opt.zero_grad();

        // Backward pass.
        loss.backward();

        // Update weights.
        opt.step();

        if epoch % 10 == 0 {
            println!("Epoch {} Loss: {:.4}", epoch, loss.double_value(&[]));
        }
    }
    
    // Save the trained model (using non-.pt extension to avoid JIT format)
    println!("Saving model to {}...", model_save_path);
    vs.save(model_save_path)?;
    println!("Model saved successfully!");
    println!("Training finished!");
    Ok(())
}