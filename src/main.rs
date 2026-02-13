use mini_gpt::training;
use tch::nn; // Import nn for VarStore

fn main() {
    println!("Starting mini-gpt training...");

    // Initialize VarStore for model parameters
    let mut vs = nn::VarStore::new(tch::Device::Cpu);

    // Dummy configuration values
    let vocab_size: i64 = 10000;
    let d_model: i64 = 512;
    let num_layers: i64 = 6;
    let num_heads: i64 = 8;
    let d_ff: i64 = 2048;
    let max_len: i64 = 128;
    let batch_size: i64 = 32;
    let seq_len: i64 = 64;
    let num_epochs: i64 = 50;
    let learning_rate: f64 = 1e-4;

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
    );
    println!("mini-gpt training finished.");
}
