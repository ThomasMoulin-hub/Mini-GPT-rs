use tch::nn;
use nn::OptimizerConfig;
use crate::{model::Transformer, data::generate_random_batch}; // Import Transformer and generate_random_batch

/// Implements a basic training loop for the Transformer model.
///
/// Args:
///   vs: The `nn::VarStore` that holds the model's parameters.
///   config: A struct or tuple containing training configuration (e.g., vocab_size, d_model).
pub fn train(
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
) {
    let device = vs.device();

    // Initialize the Transformer model.
    let model = Transformer::new(
        &vs.root(), vocab_size, d_model, num_layers, num_heads, d_ff, max_len,
    );

    // Initialize the Adam optimizer.
    // The optimizer manages and updates the model's parameters (weights and biases).
    let mut opt = nn::Adam::default().build(vs, learning_rate).expect("Failed to build optimizer");

    println!("Starting training for {} epochs...", num_epochs);

    for epoch in 1..=num_epochs {
        // Generate a random batch of data.
        // `inputs` are the token IDs for the model.
        // `targets` are the next token IDs that the model should predict.
        // We simulate a language modeling task where the model predicts the next token in the sequence.
        // For simplicity, `inputs` will be [batch_size, seq_len], and `targets` will be
        // `inputs` shifted by one position.
        let data = generate_random_batch(batch_size, seq_len + 1, vocab_size).to_device(device);
        let inputs = data.slice(1, 0, seq_len, 1); // [batch_size, seq_len]
        let targets = data.slice(1, 1, seq_len + 1, 1); // [batch_size, seq_len]

        // Forward pass: Get model predictions (logits).
        // `model.forward` expects a mask, we'll pass None for now as it's a simple example.
        // Output `logits` shape: [batch_size, seq_len, vocab_size]
        let logits = model.forward(&inputs, None);

        // Calculate the loss.
        // `cross_entropy_for_logits` expects `input` (logits) and `target`.
        // The logits need to be reshaped to `[batch_size * seq_len, vocab_size]`.
        // The targets need to be reshaped to `[batch_size * seq_len]`.
        let loss = logits
            .view([-1, vocab_size])
            .cross_entropy_for_logits(&targets.reshape([-1]));

        // Zero gradients.
        // This clears the gradients from the previous iteration, preventing accumulation.
        opt.zero_grad();

        // Backward pass.
        // Computes gradients of the loss with respect to all trainable parameters.
        loss.backward();

        // Update weights.
        // Adjusts the model's parameters based on the computed gradients and the learning rate.
        opt.step();

        if epoch % 10 == 0 {
            println!("Epoch {} Loss: {:.4}", epoch, loss.double_value(&[]));
        }
    }
    println!("Training finished!");
}