# Lesson 1: JAX Overview

## Introduction to JAX

JAX is an open-source Python library built at Google for modern ML research and production. It has become a core tool for building large language models and other neural network models.

JAX is used by many teams because it offers:
- High performance on various hardware (CPUs, GPUs, TPUs)
- Flexibility for rapid architecture iteration
- The ability to scale training across thousands of chips

Google uses JAX to build powerful models like Gemini, Gemma, Nano Banana, and Veo.

## Core Components of JAX

### 1. NumPy-Like Programming Interface

JAX's foundation is a NumPy-like interface that allows you to do complex matrix math. You can import `jax.numpy` and use familiar functions like `tanh` or `sum`, but these JAX versions can be composed with other JAX functions.

### 2. Automatic Differentiation (grad)

One of the most powerful features of JAX is automatic differentiation:
- Import `grad` from JAX
- Use it to get gradients of your functions
- JAX keeps track of all gradients automatically
- This is the key component of training neural networks
- Because JAX uses functional programming, gradient calculations are pure functions
- You can chain gradients together to do double differentiation (hard or impossible in other libraries)

### 3. Just-In-Time Compilation (jit)

JAX uses the XLA compiler (Accelerated Linear Algebra) to compile functions:
- Compile functions with the `jit` decorator or `jit()` function
- Code compiles just once on first run, then cached
- Subsequent calls run super fast
- XLA compiler can optimize for your target hardware
- Enables vectorized and parallelized execution

### 4. Vectorized Execution (vmap)

`vmap` (vectorized map) allows you to operate on batches of data:
- Import `vmap` from JAX
- Automatically vectorizes functions to accept batches
- Works seamlessly with other JAX transformations
- No need to manually write batch processing code

### 5. Hardware Sharding

Once XLA knows everything about your function and data:
- Define a sharding strategy for how JAX will split workload
- Automatically spread work across different accelerators
- Works with CPUs, GPUs, and TPUs without code rewrites
- Enables rapid experimentation at small scale, then scale to many chips

## Functional Programming in JAX

JAX uses a functional programming paradigm:
- All gradient calculations are pure functions
- Functions can be composed together
- Enables advanced features like higher-order derivatives
- Makes it hard to have side effects that could cause issues

## JAX vs. Other Libraries

### JAX vs. NumPy
- NumPy-like syntax and functionality
- JAX adds automatic differentiation
- JAX has built-in acceleration on TPUs and GPUs
- JAX functions can be composed and transformed

### JAX vs. PyTorch
- JAX is functional; PyTorch is imperative (mostly user preference)
- JAX excels at composing functions and double differentiation
- PyTorch is more familiar to many developers
- Flax/NNX provides PyTorch-like API on top of JAX

### JAX vs. TensorFlow
- JAX has more flexible programming paradigm
- No need to manually define compute graphs
- Just call jit() to build the graph automatically
- More control over function composition

## The JAX Ecosystem

The JAX ecosystem includes libraries designed to make training neural networks fast and easy:

### Core Libraries
- **JAX** — Numerical computing with autodiff, JIT, and vectorization
- **Flax/NNX** — Neural network layers and model definition (PyTorch-like syntax)
- **Grain** — Powerful data loader, especially for large datasets
- **Optax** — Optimizers and gradient processing utilities
- **Orbax** — Saving and loading model checkpoints at scale

### Advanced Libraries
- **MaxText** — For training large language models
- **MaxDiffusion** — For diffusion models
- **vLLM** — LLM inference engine with latest optimization techniques

## Key Advantages of JAX

1. **Flexibility** — Write normal Python functions and wrap them with JAX transformations
2. **Composability** — Chain multiple transformations (`grad`, `jit`, `vmap`) together
3. **Hardware Agnostic** — Same code runs on CPUs, GPUs, or TPUs without rewrites
4. **Scalability** — Easily scale from single device to thousands of chips
5. **Performance** — XLA compilation provides significant speedups
6. **Research-Friendly** — Functional paradigm enables advanced techniques like higher-order derivatives

## Why Google Uses JAX

JAX was chosen for Google's latest AI models because it:
- Offers very high performance
- Doesn't sacrifice flexibility
- Allows rapid architecture iteration
- Can efficiently use tens of thousands of chips
- Provides fine-grained control over training and serving

All of these features make JAX the foundation for Gemini, Gemma, Nano Banana, Veo, and many other cutting-edge AI models.
