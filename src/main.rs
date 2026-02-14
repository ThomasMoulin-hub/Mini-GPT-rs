use anyhow::Result;
use mini_gpt::{data, training, generation};
use tch::nn;
use std::io::{self, Write};

fn main() -> Result<()> {
    println!("=== Mini-GPT Rust ===\n");
    
    // Check if we should skip training
    let args: Vec<String> = std::env::args().collect();
    let skip_training = args.contains(&"--skip-training".to_string());
    
    // Model path - use non-.pt extension to avoid JIT format
    let model_path = "data/model_weights";
    
    // Setup data and tokenizer
    data::download_tinyshakespeare_if_needed()?;
    data::train_and_save_tokenizer_if_needed()?;
    let tokenizer = data::get_tokenizer()?;

    // Configuration values
    let vocab_size = tokenizer.get_vocab_size(true) as i64;
    let d_model: i64 = 512;
    let num_layers: i64 = 6;
    let num_heads: i64 = 8;
    let d_ff: i64 = 2048;
    let max_len: i64 = 128;
    let device = tch::Device::Cpu;
    
    if !skip_training {
        println!("Starting mini-gpt training...");
        
        // Initialize VarStore for model parameters
        let mut vs = nn::VarStore::new(device);
        
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
            model_path,
        )?;
    }
    
    // Interactive generation mode
    println!("\n=== Text Generation Mode ===");
    println!("Loading model from {}...", model_path);
    
    let (model, _vs) = generation::load_model(
        model_path,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        max_len,
        device,
    )?;
    
    println!("Model loaded successfully!\n");
    println!("Enter your prompt (or 'quit' to exit):");
    println!("Commands:");
    println!("  quit - Exit the program");
    println!("  clear - Clear screen\n");
    
    loop {
        print!("> ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input.is_empty() {
            continue;
        }
        
        if input == "quit" || input == "exit" {
            println!("Goodbye!");
            break;
        }
        
        if input == "clear" {
            print!("\x1B[2J\x1B[1;1H");
            continue;
        }
        
        // Generate text
        match generation::generate_text(&model, &tokenizer, input, 100, device, 0.8) {
            Ok(generated) => {
                println!("\nGenerated text:");
                println!("{}", generated);
                println!();
            }
            Err(e) => {
                eprintln!("Error generating text: {}", e);
            }
        }
    }

    Ok(())
}


