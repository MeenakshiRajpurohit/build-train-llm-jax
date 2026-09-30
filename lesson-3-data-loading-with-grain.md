# Lesson 3: Data Loading with Grain

## Overview

Efficiently load and preprocess text data for language model training. We cover tokenization, batching, and how to structure datasets to work with JAX's functional and parallel execution model.

## Key Concepts

### 1. Data Loading Pipeline

A complete data loading pipeline involves three essential steps:

1. **Tokenization** — Convert raw text into tokens for the model
2. **Input-Target Pairs** — Structure data so model predicts next token
3. **Batching** — Group sequences into fixed-shape arrays for efficient hardware usage

### 2. Why Grain?

**Grain** is the JAX data loading library that:
- Lazy loads data into efficient batches without overwhelming RAM
- Keeps data flowing efficiently to CPUs, GPUs, or TPUs
- Supports complex pipelines based on available hardware (single chip to thousands)
- Handles sharding across multiple machines automatically

## Implementation Steps

### Step 1: Load and Tokenize

```python
import tiktoken
from pathlib import Path

# Load raw text data
file_path = Path("TinyStories-1000.txt")
with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
    data = f.read()
    stories = data.split('<|endoftext|>')  # Split by story delimiter

print(f"Total stories: {len(stories) - 1:,}")
```

**Key Point:** Stories are separated by `<|endoftext|>` special token, which signals story completion to the model.

### Step 2: Initialize Tokenizer

```python
# Use GPT-2 tokenization scheme (standard, well-defined, open source)
tokenizer = tiktoken.get_encoding("gpt2")

print(f"Vocabulary size: {tokenizer.n_vocab:,}")      # 50,257 tokens
print(f"Special tokens: {tokenizer.special_tokens_set}")  # {'<|endoftext|>'}
```

**Vocabulary:** 50,257 tokens from GPT-2 tokenizer
**Special Tokens:** Only `<|endoftext|>` for story boundaries

### Step 3: Create Dataset Class

```python
class StoryDataset:
    def __init__(self, stories, maxlen, tokenizer):
        self.stories = stories
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.end_token = tokenizer.encode('<|endoftext|>', 
                        allowed_special={'<|endoftext|>'})[0]

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        # Tokenize the story
        tokens = self.tokenizer.encode(story, 
                                       allowed_special={'<|endoftext|>'})

        # Truncate if too long
        if len(tokens) > self.maxlen:
            tokens = tokens[:self.maxlen]
            
        # Pad with zeros if too short (model learns to ignore padding)
        tokens.extend([0] * (self.maxlen - len(tokens)))
        
        return tokens
```

**Why Padding?** 
- LLM inputs must be fixed size (128 tokens in this case)
- Zero token is used for padding (model learns to ignore it)
- No garbage data sent to model

### Step 4: Create IndexSampler (Shuffle Data)

```python
import grain.python as grain

shuffled_sampler = grain.IndexSampler(
    num_records=1000,          # Total stories
    shuffle=True,               # Randomize order (prevents overfitting)
    seed=42,                    # Reproducibility
    shard_options=grain.NoSharding(),  # Single machine (can shard to many)
    num_epochs=1                # Single pass through data
)

# Iterate through shuffled indices
for i, record_metadata in enumerate(shuffled_sampler):
    print(f"Record {i}: index={record_metadata.record_key}")
```

**Benefits:**
- Prevents training artifacts from sequential data
- `seed` ensures reproducible shuffling
- `shard_options` enables easy multi-machine training

### Step 5: Batch Data

```python
batch_op = grain.Batch(
    batch_size=32,          # Process 32 stories at a time
    drop_remainder=True     # Drop leftover stories if not divisible by 32
)
```

**Why Batching?**
- Process multiple stories simultaneously
- Better hardware utilization (GPUs/TPUs)
- Smaller batch size = lower memory usage
- Larger batch size = more efficient training (if memory allows)

### Step 6: Create Complete DataLoader

```python
def create_dataloader(
    stories,
    tokenizer,
    maxlen=128,
    batch_size=32,
    shuffle=False,
    num_epochs=1,
    seed=42,
    worker_count=0
):
    # Create dataset
    dataset = StoryDataset(stories, maxlen, tokenizer)
    estimated_batches = len(dataset) // batch_size

    # Create sampler (controls order of data)
    sampler = grain.IndexSampler(
        num_records=len(dataset),
        shuffle=shuffle,
        seed=seed,
        shard_options=grain.NoSharding(),
        num_epochs=num_epochs
    )
    
    # Create dataloader (orchestrates everything)
    dataloader = grain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[
            grain.Batch(batch_size=batch_size, drop_remainder=True)
        ],
        worker_count=worker_count  # 0 = single process
    )
    
    return dataloader, estimated_batches
```

### Step 7: Use the DataLoader

```python
# Create dataloader
dataloader, batches_per_epoch = create_dataloader(
    stories=stories,
    tokenizer=tokenizer,
    maxlen=128,
    batch_size=32,
    shuffle=True,
    num_epochs=1
)

print(f"Batches per epoch: {batches_per_epoch}")  # e.g., 3 batches

# Fetch next batch
batch = next(iter(dataloader))
print(f"Batch shape: {batch.shape}")  # (32, 128) — 32 stories, 128 tokens each
```

## Data Flow

```
Raw Text Stories
    ↓
Split by <|endoftext|>
    ↓
StoryDataset (tokenize + pad)
    ↓
IndexSampler (shuffle)
    ↓
grain.Batch (group into 32)
    ↓
DataLoader (iterator)
    ↓
Training Loop (32 × 128 token batches)
```

## Key Parameters

| Parameter | Purpose | Typical Values |
|-----------|---------|-----------------|
| `maxlen` | Max tokens per story | 128-512 |
| `batch_size` | Stories per batch | 16-128 |
| `shuffle` | Randomize order | True (training) / False (eval) |
| `num_epochs` | Passes through data | 1-5 |
| `seed` | Reproducibility | 42 |
| `drop_remainder` | Handle incomplete batches | True |
| `worker_count` | Parallel data loading | 0 (single) to CPU count |

## Performance Tips

1. **Batch Size:** Start small (32), increase if memory allows
2. **Workers:** Use `worker_count > 0` for large datasets to parallelize I/O
3. **Shuffling:** Always shuffle training data to prevent overfitting
4. **Maxlen:** Balance between capturing full stories and memory usage
5. **Epochs:** 1-3 usually sufficient for small datasets

## Next Steps

- This pipeline feeds directly into the training loop
- Model receives batches of shape (batch_size, maxlen)
- Trainer uses these to predict next token and compute loss
- Grain handles distributed training seamlessly
