use anyhow::Result;
use tch::{nn, Device, Tensor};
use nn::OptimizerConfig;
use crate::model::Transformer;

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
) -> Result<()> {
    let device = vs.device();

    // Initialize the Transformer model.
    let model = Transformer::new(
        &vs.root(), vocab_size, d_model, num_layers, num_heads, d_ff, max_len,
    );

    // Initialize the Adam optimizer.
    let mut opt = nn::Adam::default().build(vs, learning_rate)?;

    println!("Starting training for {} epochs...", num_epochs);

    for epoch in 1..=num_epochs {
        // Generate a batch of data using the provided closure.
        let (inputs, targets) = batch_generator(batch_size, seq_len, device)?;

        // Forward pass: Get model predictions (logits).
        let logits = model.forward(&inputs, None);

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
    println!("Training finished!");
    Ok(())
}