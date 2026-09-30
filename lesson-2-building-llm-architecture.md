# Lesson 2: Building the LLM Architecture

## Overview

In this lesson, you'll build the complete MiniGPT model architecture using JAX and Flax/NNX. You'll implement:
- Token and position embeddings
- Causal attention masking
- Transformer blocks with multi-head attention
- The complete MiniGPT model with 20 million parameters

## Creating Embedding Layers

### TokenAndPositionEmbedding Class

Embeddings convert discrete tokens (integers) into dense vector representations. We need two types:

**Token Embedding**: Maps each token ID to a learned embedding vector
**Position Embedding**: Encodes the position of each token in the sequence

```python
class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, *, rngs):
        # Token embeddings: map token IDs to vectors
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        # Position embeddings: encode position in sequence
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

    def __call__(self, x):
        seq_len = x.shape[1]
        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = jnp.arange(seq_len)[None, :]
        # Add token and position embeddings
        return self.token_emb(x) + self.pos_emb(positions)
```

**Key Points:**
- `vocab_size`: Number of unique tokens in vocabulary (typically 50,000+ for language models)
- `embed_dim`: Dimension of embedding vectors (e.g., 192 for MiniGPT)
- `maxlen`: Maximum sequence length (e.g., 128 tokens)
- Both embeddings are added together to get final token representation with positional information

## Causal Attention Masking

In language models, we want to prevent the model from looking at future tokens during training. This is enforced with a **causal attention mask**.

```python
def causal_attention_mask(seq_len):
    return jnp.tril(jnp.ones((seq_len, seq_len)))
```

**What it does:**
- Creates a lower triangular matrix (1s below/on diagonal, 0s above)
- Prevents attention from position i to positions j > i
- Ensures autoregressive generation (each token only depends on previous tokens)

**Example for seq_len=8:**
```
Position: 0 1 2 3 4 5 6 7
      0: 1 0 0 0 0 0 0 0  (token 0 can only attend to itself)
      1: 1 1 0 0 0 0 0 0  (token 1 can attend to 0,1)
      2: 1 1 1 0 0 0 0 0  (token 2 can attend to 0,1,2)
      3: 1 1 1 1 0 0 0 0  (token 3 can attend to 0,1,2,3)
      ...
      7: 1 1 1 1 1 1 1 1  (token 7 can attend to all previous tokens)
```

## Building the Transformer Block

A Transformer block consists of:
1. Multi-head self-attention
2. Residual connection (skip connection)
3. Feed-forward network (in a full implementation)

```python
class TransformerBlock(nnx.Module):

    def __init__(self, embed_dim, num_heads, ff_dim, *, rngs):
        # Multi-head attention layer
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            decode=False,  # We're in training mode, not generation
            rngs=rngs
        )
        
    def __call__(self, x, mask=None):
        # Apply multi-head attention
        attn_out = self.attention(x, mask=mask)
        # Residual connection: add input to output
        x = x + attn_out
        return x
```

**Components:**
- `num_heads`: Number of attention heads (6 for MiniGPT)
- `in_features`: Input dimension (embed_dim)
- `qkv_features`: Dimension of Query, Key, Value projections
- `mask`: Causal mask to prevent attending to future tokens
- Residual connection helps with gradient flow during backpropagation

## Complete MiniGPT Model

Now we combine all components into the full model:

```python
class MiniGPT(nnx.Module):

    def __init__(self, maxlen, vocab_size, embed_dim, num_heads,
                 feed_forward_dim, num_transformer_blocks, *, rngs):
        self.maxlen = maxlen
        
        # Embedding layer (token + position)
        self.embedding = TokenAndPositionEmbedding(maxlen, vocab_size, 
                                                   embed_dim, rngs=rngs)
        
        # Stack of transformer blocks
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs)
            for _ in range(num_transformer_blocks)
        ]
        
        # Output layer: project to vocabulary size
        self.output_layer = nnx.Linear(embed_dim, vocab_size, 
                                       use_bias=False, rngs=rngs)
        
    def causal_attention_mask(self, seq_len):
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def __call__(self, token_ids):
        # Get sequence length from input
        seq_len = token_ids.shape[1]
        # Create causal attention mask
        mask = self.causal_attention_mask(seq_len)
        # Pass through embedding layer
        x = self.embedding(token_ids)
        # Pass through all transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask=mask)
        # Project to vocabulary size to get logits
        logits = self.output_layer(x)
        return logits
```

## Instantiating the Model

```python
import tiktoken
# Load GPT-2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
print(f"Vocabulary size: {tokenizer.n_vocab}")  # Output: 50257

# Create MiniGPT model
model = MiniGPT(
    maxlen=128,                    # Max sequence length
    vocab_size=tokenizer.n_vocab,  # 50,257 tokens
    embed_dim=192,                 # Embedding dimension
    num_heads=6,                   # Number of attention heads
    feed_forward_dim=512,          # Feed-forward hidden dimension
    num_transformer_blocks=6,      # Number of transformer blocks
    rngs=nnx.Rngs(0)              # Random number generator with seed 0
)
```

## Model Architecture Summary

**Parameters by Component:**
- Embeddings: 
  - Token: vocab_size × embed_dim = 50,257 × 192
  - Position: maxlen × embed_dim = 128 × 192
- Transformer Blocks (×6):
  - Multi-head attention: ~embed_dim² parameters per head
  - Total across 6 blocks
- Output layer: embed_dim × vocab_size = 192 × 50,257

**Total Parameters:** ~20 million

## Key Concepts

1. **Embeddings**: Convert discrete tokens to continuous vectors, with positional information
2. **Causal Masking**: Ensures model doesn't cheat by looking at future tokens
3. **Multi-Head Attention**: Allows model to attend to different representations of input
4. **Transformer Blocks**: Repeated layers of attention and processing
5. **Residual Connections**: Help with training deeper networks
6. **Output Layer**: Projects learned representations back to vocabulary size for next-token prediction

## Next Steps

Now that you have the model architecture defined, the next lesson will cover:
- How to prepare training data
- Implementing the training loop
- Using Optax for optimization
- Monitoring loss and saving checkpoints
