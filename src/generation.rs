use anyhow::Result;
use tch::{nn, Device, Tensor, Kind};
use tokenizers::Tokenizer;
use crate::model::Transformer;

/// Generates text from a prompt using a trained Transformer model.
///
/// Args:
///   model: The trained Transformer model.
///   tokenizer: The tokenizer to encode/decode text.
///   prompt: The input text to start generation from.
///   max_new_tokens: Maximum number of tokens to generate.
///   device: The device to run inference on.
///   temperature: Sampling temperature (higher = more random, lower = more deterministic).
///
/// Returns:
///   The generated text including the prompt.
pub fn generate_text(
    model: &Transformer,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    device: Device,
    temperature: f64,
) -> Result<String> {
    // Encode the prompt
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e: tokenizers::Error| anyhow::anyhow!(e))?;
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

    if input_ids.is_empty() {
        return Err(anyhow::anyhow!("Empty prompt after tokenization"));
    }

    // Convert to tensor: shape [1, seq_len]
    let mut context = Tensor::from_slice(&input_ids)
        .to(device)
        .unsqueeze(0);

    println!("Generating {} tokens...", max_new_tokens);

    // Generate tokens one by one
    for i in 0..max_new_tokens {
        // Get the sequence length
        let seq_len = context.size()[1];

        // Create causal mask for this sequence length
        let causal_mask = create_causal_mask(seq_len, device);

        // Forward pass - get logits for all positions
        // logits shape: [1, seq_len, vocab_size]
        let logits = tch::no_grad(|| {
            model.forward(&context, Some(&causal_mask))
        });

        // Get logits for the last position: [1, vocab_size]
        // Use select to get the last token along dimension 1
        let last_logits = logits.select(1, seq_len - 1);

        // Apply temperature scaling
        let scaled_logits = last_logits / temperature;

        // Sample from the distribution
        let probs = scaled_logits.softmax(-1, Kind::Float);
        let next_token = probs.multinomial(1, false);

        // Append the new token to the context
        context = Tensor::cat(&[context, next_token], 1);

        // Optional: print progress
        if (i + 1) % 10 == 0 {
            print!(".");
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }

    println!(); // New line after generation

    // Decode the generated sequence
    // Select the first (and only) batch item
    let sequence = context.select(0, 0).to(Device::Cpu);

    // Convert tensor to Vec<u32>
    let seq_len = sequence.size()[0] as usize;
    let mut generated_ids = Vec::with_capacity(seq_len);

    for i in 0..seq_len {
        let val: i64 = sequence.int64_value(&[i as i64]);
        generated_ids.push(val as u32);
    }

    let generated_text = tokenizer.decode(&generated_ids, true)
        .map_err(|e: tokenizers::Error| anyhow::anyhow!(e))?;

    Ok(generated_text)
}

/// Creates a causal mask for decoder self-attention.
fn create_causal_mask(seq_len: i64, device: Device) -> Tensor {
    let ones = Tensor::ones(&[seq_len, seq_len], (Kind::Float, device));
    let mask = ones.tril(0);
    let large_negative = -1e9_f64;
    let mask = (&mask - 1.0) * large_negative;
    mask.unsqueeze(0).unsqueeze(0)
}

/// Loads a trained model from a file.
///
/// Args:
///   model_path: Path to the saved model weights.
///   vocab_size: Vocabulary size.
///   d_model: Model dimension.
///   num_layers: Number of transformer layers.
///   num_heads: Number of attention heads.
///   d_ff: Feed-forward dimension.
///   max_len: Maximum sequence length.
///   device: Device to load the model on.
///
/// Returns:
///   The loaded Transformer model and VarStore.
pub fn load_model(
    model_path: &str,
    vocab_size: i64,
    d_model: i64,
    num_layers: i64,
    num_heads: i64,
    d_ff: i64,
    max_len: i64,
    device: Device,
) -> Result<(Transformer, nn::VarStore)> {
    let mut vs = nn::VarStore::new(device);
    let model = Transformer::new(
        &vs.root(),
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        max_len,
    );

    // Load the saved weights
    vs.load(model_path)?;

    Ok((model, vs))
}

