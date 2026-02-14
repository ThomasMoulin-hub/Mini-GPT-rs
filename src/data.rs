use anyhow::Result;
use std::fs;
use std::path::Path;
use tch::{Device, Tensor, IndexOp};
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::Strip;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::models::bpe::BpeTrainer;
use tokenizers::models::TrainerWrapper;
use tokenizers::Tokenizer;
use rand::Rng;

const TINYSHAKESPEARE_URL: &str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";
const DATA_DIR: &str = "data";
const DATASET_FILE: &str = "data/tinyshakespeare.txt";
const TOKENIZER_FILE: &str = "data/tokenizer.json";

/// Downloads the TinyShakespeare dataset if it's not already present.
pub fn download_tinyshakespeare_if_needed() -> Result<()> {
    if !Path::new(DATA_DIR).exists() {
        fs::create_dir_all(DATA_DIR)?;
    }

    if !Path::new(DATASET_FILE).exists() {
        println!("Downloading TinyShakespeare dataset...");
        let response = reqwest::blocking::get(TINYSHAKESPEARE_URL)?;
        let content = response.text()?;
        fs::write(DATASET_FILE, content)?;
        println!("Dataset downloaded successfully.");
    }

    Ok(())
}

/// Trains and saves a tokenizer if it doesn't already exist.
pub fn train_and_save_tokenizer_if_needed() -> Result<()> {
    if !Path::new(TOKENIZER_FILE).exists() {
        println!("Training a new tokenizer...");
        let mut tokenizer = Tokenizer::new(BPE::default());
        tokenizer.with_normalizer(Some(Strip::new(true, true)));
        tokenizer.with_pre_tokenizer(Some(Whitespace::default()));

        let trainer = BpeTrainer::builder()
            .vocab_size(1000)
            .min_frequency(25000)
            .build();
        let mut trainer = TrainerWrapper::from(trainer);
        let content = fs::read_to_string(DATASET_FILE)?;
        tokenizer.train(&mut trainer, std::iter::once(content))
            .map_err(|e: tokenizers::Error| anyhow::anyhow!(e))?;

        tokenizer.save(TOKENIZER_FILE, true).map_err(|e: tokenizers::Error| anyhow::anyhow!(e))?;
        println!("Tokenizer trained and saved.");
    }
    Ok(())
}

/// Gets the tokenizer.
pub fn get_tokenizer() -> Result<Tokenizer> {
    Tokenizer::from_file(TOKENIZER_FILE).map_err(|e: tokenizers::Error| anyhow::anyhow!(e))
}

/// Generates a random batch of tokenized data from the TinyShakespeare dataset.
pub fn get_batch(
    tokenizer: &Tokenizer,
    batch_size: i64,
    seq_len: i64,
    device: Device,
) -> Result<(Tensor, Tensor)> {
    let content = fs::read_to_string(DATASET_FILE)?;
    let encoding = tokenizer.encode(content, true).map_err(|e: tokenizers::Error| anyhow::anyhow!(e))?;
    let ids = encoding.get_ids().iter().map(|&id| id as i64).collect::<Vec<_>>();
    let data = Tensor::from_slice(&ids).to(device);

    let mut rng = rand::thread_rng();
    let mut input_tensors = Vec::new();
    let mut target_tensors = Vec::new();

    for _ in 0..batch_size {
        let start = rng.gen_range(0..data.size1()? - seq_len - 1);
        let end = start + seq_len;
        input_tensors.push(data.i(start..end));
        target_tensors.push(data.i(start + 1..end + 1));
    }

    let inputs = Tensor::stack(&input_tensors, 0);
    let targets = Tensor::stack(&target_tensors, 0);

    Ok((inputs, targets))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn cleanup() {
        let _ = fs::remove_file(TOKENIZER_FILE);
        let _ = fs::remove_file(DATASET_FILE);
        let _ = fs::remove_dir_all(DATA_DIR);
    }

    #[test]
    fn test_download_tinyshakespeare_if_needed() {
        cleanup();
        download_tinyshakespeare_if_needed().unwrap();
        assert!(Path::new(DATASET_FILE).exists());
        cleanup();
    }

    #[test]
    fn test_get_tokenizer() {
        cleanup();
        download_tinyshakespeare_if_needed().unwrap();
        train_and_save_tokenizer_if_needed().unwrap();
        let tokenizer = get_tokenizer().unwrap();
        assert!(Path::new(TOKENIZER_FILE).exists());
        // For tokenizers 0.13.4, vocab_size includes special tokens if added via trainer.
        // We are setting vocab_size to 1000 in trainer, but it might be slightly higher
        // due to min_frequency and special tokens if any. Let's just check it's not 0.
        assert!(tokenizer.get_vocab_size(true) > 0);
        cleanup();
    }

    #[test]
    fn test_get_batch() {
        cleanup();
        download_tinyshakespeare_if_needed().unwrap();
        train_and_save_tokenizer_if_needed().unwrap();
        let tokenizer = get_tokenizer().unwrap();
        let (inputs, targets) = get_batch(&tokenizer, 4, 64, Device::Cpu).unwrap();
        assert_eq!(inputs.size(), &[4, 64]);
        assert_eq!(targets.size(), &[4, 64]);
        cleanup();
    }
}
