# Build and Train an LLM with JAX

A short course, built in partnership with Google and taught by Chris Achard (Developer Relations Engineer on Google's TPU Software team), on building and training a MiniGPT-style language model from scratch using JAX.

## What you'll learn

- Build a GPT-2 style language model with 20 million parameters from scratch using JAX, the open-source library behind Google's Gemini, Veo, and Nano Banana models.
- Learn JAX's core primitives (automatic differentiation, JIT compilation, and vectorized mapping) and how to combine them to define, train, and checkpoint a neural network efficiently.
- Load a pretrained MiniGPT model and run inference through a chat interface, completing the full workflow from data preprocessing and training to generating text with the trained LLM.

## About this course

JAX is the open-source numerical computing library that Google uses to build and train its most advanced models, including Gemini. It looks similar to NumPy, but adds automatic differentiation, just-in-time compilation, and the ability to scale training across thousands of CPUs, GPUs, and TPUs. In this course, you'll learn JAX by building and training a language model from scratch.

You'll implement a complete MiniGPT-style LLM with 20 million parameters — defining the architecture, loading and preprocessing training data, running the training loop, saving checkpoints, and finally chatting with your trained model through a graphical interface. Along the way, you'll work with key tools from the JAX ecosystem: Flax/NNX for neural network layers, Grain for data loading, Optax for optimization, and Orbax for checkpointing.

In detail, you'll:

1. Explore JAX's core concepts — automatic differentiation, JIT compilation, and vectorized execution — and see how it compares to NumPy, PyTorch, and TensorFlow in the broader ML landscape.
2. Build the architecture of a MiniGPT-style language model using JAX and Flax/NNX, implementing token embeddings and transformer blocks into a complete, trainable model.
3. Load and preprocess a dataset of mini stories for training, covering tokenization, batching, and structuring data for JAX's functional execution model.
4. Implement the full training loop — computing losses, applying gradients with Optax, and using JAX transformations to keep training efficient — then save your model with Orbax checkpointing.
5. Load a pretrained MiniGPT model and run inference through a chat interface to generate stories, completing the full build-train-deploy workflow.

The steps you'll follow to build and train MiniGPT are the same foundational steps Google uses to develop its more powerful models like Gemini. This course gives you hands-on experience with the tools and techniques at the core of modern LLM development.

## Tools & Ecosystem

- **JAX** — core numerical computing, autodiff, JIT compilation, vectorization
- **Flax/NNX** — neural network layers and model definition
- **Grain** — data loading
- **Optax** — optimization
- **Orbax** — checkpointing
