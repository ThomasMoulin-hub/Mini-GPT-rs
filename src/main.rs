use anyhow::Result;
use mini_gpt::{data, training};
use tch::nn;

fn main() -> Result<()> {
    println!("Starting mini-gpt training...");

    // Setup data and tokenizer
    data::download_tinyshakespeare_if_needed()?;
    data::train_and_save_tokenizer_if_needed()?;
    let tokenizer = data::get_tokenizer()?;

    // Initialize VarStore for model parameters
    let mut vs = nn::VarStore::new(tch::Device::Cpu);

    // Configuration values
    let vocab_size = tokenizer.get_vocab_size(true) as i64;
    let d_model: i64 = 512;
    let num_layers: i64 = 6;
    let num_heads: i64 = 8;
    let d_ff: i64 = 2048;
    let max_len: i64 = 128;
    let batch_size: i64 = 32;
    let seq_len: i64 = 64;
    let num_epochs: i64 = 50;
    let learning_rate: f64 = 1e-4;

    // Pass a closure to the train function for generating batches
    let batch_generator = |batch_size, seq_len, device| {
        data::get_batch(&tokenizer, batch_size, seq_len, device)
    };

    training::train(
        &mut vs,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        max_len,
        batch_size,
        seq_len,
        num_epochs,
        learning_rate,
        &batch_generator,
    )?;

    println!("mini-gpt training finished.");
    Ok(())
}
