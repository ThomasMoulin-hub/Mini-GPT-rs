use tch::{nn, Tensor};
use std::borrow::Borrow;

use crate::layers::{PositionalEncoding, TransformerBlock}; // Import layers

#[derive(Debug)]
pub struct Transformer {
    token_embeddings: nn::Embedding,
    pos_encoder: PositionalEncoding,
    blocks: Vec<TransformerBlock>,
    norm: nn::LayerNorm,
    lm_head: nn::Linear,
    d_model: i64,
    vocab_size: i64,
    max_len: i64,
}

impl Transformer {
    /// Creates a new Transformer (Decoder-only GPT-style) model.
    ///
    /// Args:
    ///   vs: The `nn::Path` for `VarStore` ownership.
    ///   vocab_size: The size of the vocabulary.
    ///   d_model: The dimensionality of the input and output embeddings.
    ///   num_layers: The number of Transformer blocks.
    ///   num_heads: The number of attention heads in MultiHeadAttention.
    ///   d_ff: The dimensionality of the inner feed-forward layer.
    ///   max_len: The maximum sequence length for positional encoding.
    pub fn new<'a, P: Borrow<nn::Path<'a>>>(
        vs: P,
        vocab_size: i64,
        d_model: i64,
        num_layers: i64,
        num_heads: i64,
        d_ff: i64,
        max_len: i64,
    ) -> Self {
        let vs = vs.borrow(); // Borrow the nn::Path

        // Token embeddings layer.
        // It maps token IDs to d_model-dimensional vectors.
        let token_embeddings = nn::embedding(
            vs / "token_embeddings",
            vocab_size,
            d_model,
            Default::default(),
        );

        // Positional encoding layer.
        // Adds positional information to the token embeddings.
        let pos_encoder = PositionalEncoding::new(d_model, max_len);

        // Stack of Transformer blocks.
        // Each block consists of multi-head attention and a feed-forward network.
        let blocks: Vec<TransformerBlock> = (0..num_layers)
            .map(|i| TransformerBlock::new(vs / format!("block{}", i), d_model, num_heads, d_ff))
            .collect();

        // Final Layer Normalization after the stack of blocks.
        let norm = nn::layer_norm(vs / "norm", vec![d_model], Default::default());

        // Language modeling head (linear layer).
        // Projects the output of the Transformer to logits over the vocabulary.
        let lm_head = nn::linear(vs / "lm_head", d_model, vocab_size, Default::default());

        Self {
            token_embeddings,
            pos_encoder,
            blocks,
            norm,
            lm_head,
            d_model,
            vocab_size,
            max_len,
        }
    }

    /// Performs the forward pass of the Transformer model.
    ///
    /// Args:
    ///   xs: The input tensor of token IDs, shape `[batch_size, seq_len]`.
    ///   mask: Optional attention mask, shape `[batch_size, 1, seq_len, seq_len]`.
    ///
    /// Returns:
    ///   The output tensor of logits, shape `[batch_size, seq_len, vocab_size]`.
    pub fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // 1. Token embeddings.
        // Input `xs` (token IDs) shape: [batch_size, seq_len]
        // Output `x` shape: [batch_size, seq_len, d_model]
        let mut x = xs.apply(&self.token_embeddings);

        // 2. Add positional encoding.
        // `x` shape remains: [batch_size, seq_len, d_model]
        x = self.pos_encoder.forward(&x);

        // 3. Pass through Transformer blocks.
        // `x` shape remains: [batch_size, seq_len, d_model]
        for block in self.blocks.iter() {
            x = block.forward(&x, mask);
        }

        // 4. Final Layer Normalization.
        // `x` shape remains: [batch_size, seq_len, d_model]
        x = x.apply(&self.norm);

        // 5. Language modeling head (output logits).
        // Output `x` shape: [batch_size, seq_len, vocab_size]
        x.apply(&self.lm_head)
    }
}
