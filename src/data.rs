use tch::Tensor;

/// Generates a random batch of tokenized data.
///
/// Args:
///   batch_size: The number of sequences in the batch.
///   seq_len: The length of each sequence.
///   vocab_size: The size of the vocabulary.
///
/// Returns:
///   A tensor of shape `[batch_size, seq_len]` containing random token IDs.
pub fn generate_random_batch(batch_size: i64, seq_len: i64, vocab_size: i64) -> Tensor {
    // Generates a tensor of random integers (token IDs) between 0 and vocab_size - 1.
    // The `rand_int` function creates a tensor with values in the range [low, high).
    Tensor::randint(vocab_size, &[batch_size, seq_len], (tch::kind::Kind::Int64, tch::Device::Cpu))
}
