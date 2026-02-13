use tch::{nn, Tensor};
use std::borrow::Borrow;

#[derive(Debug)]
pub struct PositionalEncoding {
    pe: Tensor,
}

impl PositionalEncoding {
    /// Creates a new PositionalEncoding layer.
    ///
    /// Args:
    ///   d_model: The dimensionality of the input embeddings.
    ///   max_len: The maximum sequence length.
    pub fn new(d_model: i64, max_len: i64) -> Self {
        // Initialize a zero tensor for positional encodings.
        // Shape: [max_len, d_model]
        let pe = Tensor::zeros(&[max_len, d_model], (tch::kind::Kind::Float, tch::Device::Cpu));

        // Create a tensor for positions: [0, 1, ..., max_len - 1].
        // Shape: [max_len, 1]
        let position = Tensor::arange(max_len, (tch::kind::Kind::Float, tch::Device::Cpu)).unsqueeze(-1);

        // Create a tensor for div_term: 1 / (10000^(2i/d_model)).
        // Shape: [d_model / 2]
        // This calculates the inverse of 10000 raised to the power of
        // 2 times each dimension index divided by d_model.
        // The division by 2 in d_model/2 is because we apply sine to
        // even indices and cosine to odd indices, effectively halving
        // the frequency calculation per dimension.
        let factor = Tensor::arange_start_step(0, d_model, 2, (tch::kind::Kind::Float, tch::Device::Cpu))
            * -(Tensor::from(10000.0).log() / d_model as f64);
        let div_term = factor.exp();

        // Apply sine to even indices.
        // `position` has shape [max_len, 1], `div_term` has shape [d_model/2].
        // When performing `position * div_term`, `position` is broadcasted across `d_model/2`
        // and `div_term` is broadcasted across `max_len`.
        // The resulting tensor has shape [max_len, d_model/2].
        pe.slice(1, 0, d_model, 2)
            .copy_(&((&position * &div_term).sin()));

        // Apply cosine to odd indices.
        // Similar broadcasting occurs here.
        pe.slice(1, 1, d_model, 2)
            .copy_(&((&position * &div_term).cos()));

        // `pe` is now `[max_len, d_model]`. For actual usage, we usually
        // unsqueeze it to `[1, max_len, d_model]` to make it broadcastable
        // with batch-first input tensors of shape `[batch_size, seq_len, d_model]`.
        // This is done in the `forward` method.
        Self { pe }
    }

    /// Applies positional encoding to the input tensor.
    ///
    /// Args:
    ///   xs: The input tensor, typically of shape `[batch_size, seq_len, d_model]`.
    ///
    /// Returns:
    ///   The input tensor with positional encoding added.
    pub fn forward(&self, xs: &Tensor) -> Tensor {
        // Ensure the positional encoding tensor is ready for broadcasting.
        // If pe is [max_len, d_model], make it [1, max_len, d_model].
        // This allows it to be added to `xs` (e.g., [batch_size, seq_len, d_model])
        // via broadcasting, as `pe` will be expanded to match the batch_size.
        // The slice `pe.narrow(0, 0, xs.size()[1])` takes the first `seq_len`
        // elements along dimension 0 (the max_len dimension).
        let seq_len = xs.size()[1];
        xs + self.pe.narrow(0, 0, seq_len).unsqueeze(0)
    }
}


#[derive(Debug)]
pub struct MultiHeadAttention {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
    num_heads: i64,
    head_dim: i64,
    scaling_factor: f64,
}

impl MultiHeadAttention {
    /// Creates a new MultiHeadAttention layer.
    ///
    /// Args:
    ///   vs: The `nn::Path` for `VarStore` ownership, used to create trainable layers.
    ///   d_model: The dimensionality of the input and output.
    ///   num_heads: The number of attention heads.
    pub fn new<'a, P: Borrow<nn::Path<'a>>>(vs: P, d_model: i64, num_heads: i64) -> Self {
        let vs = vs.borrow(); // Borrow the nn::Path
        let head_dim = d_model / num_heads;
        if d_model % num_heads != 0 {
            panic!("d_model must be divisible by num_heads");
        }

        // Initialize linear projection layers for query, key, value, and output.
        // Each layer takes `d_model` as input and produces `d_model` as output.
        // `d_model` is effectively `num_heads * head_dim`.
        // These layers own their weights (parameters) through the `vs` (VarStore path).
        let q_proj = nn::linear(vs / "q_proj", d_model, d_model, Default::default());
        let k_proj = nn::linear(vs / "k_proj", d_model, d_model, Default::default());
        let v_proj = nn::linear(vs / "v_proj", d_model, d_model, Default::default());
        let out_proj = nn::linear(vs / "out_proj", d_model, d_model, Default::default());

        // Scaling factor for scaled dot-product attention: sqrt(head_dim).
        // This is used to prevent the dot products from becoming too large,
        // which can push the softmax function into regions with very small gradients.
        let scaling_factor = (head_dim as f64).sqrt();

        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scaling_factor,
        }
    }

    /// Performs the multi-head attention forward pass.
    ///
    /// Args:
    ///   query: The query tensor, shape `[batch_size, seq_len, d_model]`.
    ///   key: The key tensor, shape `[batch_size, seq_len, d_model]`.
    ///   value: The value tensor, shape `[batch_size, seq_len, d_model]`.
    ///   mask: Optional attention mask, shape `[batch_size, 1, seq_len, seq_len]` or `[1, 1, seq_len, seq_len]`.
    ///         Typically used for decoder self-attention to prevent attending to future tokens.
    ///
    /// Returns:
    ///   The output tensor after multi-head attention, shape `[batch_size, seq_len, d_model]`.
    pub fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let (batch_size, seq_len, d_model) = query.size3().expect("query must have 3 dimensions");

        // 1. Linear projections for Q, K, V.
        // Resulting shape for q, k, v: [batch_size, seq_len, d_model]
        let q = query.apply(&self.q_proj);
        let k = key.apply(&self.k_proj);
        let v = value.apply(&self.v_proj);

        // 2. Reshape Q, K, V for multiple heads.
        // From [batch_size, seq_len, d_model] to [batch_size, seq_len, num_heads, head_dim]
        // Then transpose to [batch_size, num_heads, seq_len, head_dim]
        // This reordering is crucial for efficient batch matrix multiplication.
        let q = q
            .view([batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let k = k
            .view([batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let v = v
            .view([batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);

        // 3. Scaled Dot-Product Attention.
        // (Q @ K.transpose) / sqrt(head_dim)
        // q has shape [batch_size, num_heads, seq_len, head_dim]
        // k.transpose has shape [batch_size, num_heads, head_dim, seq_len]
        // scores will have shape [batch_size, num_heads, seq_len, seq_len]
        let scores = q.matmul(&k.transpose(-2, -1)) / Tensor::from(self.scaling_factor);

        // Apply optional mask (e.g., for decoder self-attention).
        // Masking typically involves setting scores of future tokens to a very small negative number
        // so that their softmax probability becomes zero.
        let scores = if let Some(m) = mask {
            scores + m
        } else {
            scores
        };

        // Apply softmax to get attention probabilities.
        // Shape: [batch_size, num_heads, seq_len, seq_len]
        let attn_weights = scores.softmax(-1, tch::kind::Kind::Float);

        // Multiply attention weights with values.
        // attn_weights has shape [batch_size, num_heads, seq_len, seq_len]
        // v has shape [batch_size, num_heads, seq_len, head_dim]
        // attn_output will have shape [batch_size, num_heads, seq_len, head_dim]
        let attn_output = attn_weights.matmul(&v);

        // 4. Concatenate heads and apply final linear layer.
        // Transpose back to [batch_size, seq_len, num_heads, head_dim]
        // Then reshape to [batch_size, seq_len, d_model]
        let attn_output = attn_output
            .transpose(1, 2)
            .contiguous() // Ensure memory is contiguous after transpose for view operation.
            .view([batch_size, seq_len, d_model]);

        // Apply final output projection.
        attn_output.apply(&self.out_proj)
    }
}

#[derive(Debug)]
pub struct FeedForward {
    linear1: nn::Linear,
    linear2: nn::Linear,
}

impl FeedForward {
    /// Creates a new FeedForward layer.
    ///
    /// Args:
    ///   vs: The `nn::Path` for `VarStore` ownership.
    ///   d_model: The dimensionality of the input and output.
    ///   d_ff: The dimensionality of the inner feed-forward layer.
    pub fn new<'a, P: Borrow<nn::Path<'a>>>(vs: P, d_model: i64, d_ff: i64) -> Self {
        let vs = vs.borrow(); // Borrow the nn::Path
        
        // The first linear layer expands the dimension from d_model to d_ff.
        let linear1 = nn::linear(vs / "linear1", d_model, d_ff, Default::default());
        // The second linear layer projects it back from d_ff to d_model.
        let linear2 = nn::linear(vs / "linear2", d_ff, d_model, Default::default());

        Self { linear1, linear2 }
    }

    /// Performs the forward pass for the FeedForward layer.
    ///
    /// Args:
    ///   xs: The input tensor, typically of shape `[batch_size, seq_len, d_model]`.
    ///
    /// Returns:
    ///   The output tensor after passing through the feed-forward network,
    ///   shape `[batch_size, seq_len, d_model]`.
    pub fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.linear1).relu().apply(&self.linear2)
    }
}

#[derive(Debug)]
pub struct TransformerBlock {
    self_attn: MultiHeadAttention,
    ff: FeedForward,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
}

impl TransformerBlock {
    /// Creates a new TransformerBlock layer.
    ///
    /// Args:
    ///   vs: The `nn::Path` for `VarStore` ownership.
    ///   d_model: The dimensionality of the input and output.
    ///   num_heads: The number of attention heads.
    ///   d_ff: The dimensionality of the inner feed-forward layer.
    pub fn new<'a, P: Borrow<nn::Path<'a>>>(vs: P, d_model: i64, num_heads: i64, d_ff: i64) -> Self {
        let vs = vs.borrow(); // Borrow the nn::Path
        
        let self_attn = MultiHeadAttention::new(vs / "self_attn", d_model, num_heads);
        let ff = FeedForward::new(vs / "ff", d_model, d_ff);
        // nn::layer_norm will create weights and biases using the provided nn::Path
        let norm1 = nn::layer_norm(vs / "norm1", vec![d_model], Default::default());
        let norm2 = nn::layer_norm(vs / "norm2", vec![d_model], Default::default());

        Self { self_attn, ff, norm1, norm2 }
    }

    /// Performs the forward pass for the TransformerBlock.
    ///
    /// Args:
    ///   xs: The input tensor, typically of shape `[batch_size, seq_len, d_model]`.
    ///   mask: Optional attention mask.
    ///
    /// Returns:
    ///   The output tensor after passing through the TransformerBlock.
    pub fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // Multi-Head Attention with residual connection and LayerNorm.
        let attn_output = self.self_attn.forward(&xs, &xs, &xs, mask);
        let xs_after_attn = xs + &attn_output.apply(&self.norm1); // Add & Norm

        // Feed-Forward with residual connection and LayerNorm.
        let ff_output = self.ff.forward(&xs_after_attn);
        xs_after_attn + &ff_output.apply(&self.norm2) // Add & Norm
    }
}