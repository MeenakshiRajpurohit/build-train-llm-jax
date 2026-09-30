# Introduction — Build and Train an LLM with JAX

## Course Overview

This course teaches you how to build and train a mini GPT model from scratch using JAX.

## What is JAX?

JAX is an open-source Python library from Google for fast numerical computing and machine learning. It looks similar to NumPy, but includes additional key features optimized for machine learning:

- **Automatic differentiation** — Automatically compute gradients for neural networks
- **Just-in-time compilation** — Compile code for speed using the XLA compiler
- **Multi-device support** — Run efficiently on CPUs, GPUs, or TPUs without code rewrites
- **Hardware scalability** — Scale training from single device to thousands of chips

## Why JAX?

JAX is used by many teams at Google and elsewhere because:
- It's the popular choice for researchers who need control over hardware execution
- It enables both high performance and flexibility
- It supports rapid architecture iteration on large-scale hardware
- It powers Google's most advanced models: Gemini, Gemma, Nano Banana, and Veo

## JAX Core Primitives

JAX allows you to write normal Python functions and wrap them with special transformation tools:

- **`grad`** — Automatic differentiation for computing gradients
- **`jit`** — Just-in-time compilation using XLA (Accelerated Linear Algebra) compiler
- **`vmap`** — Vectorized mapping to run code over many inputs at once

These primitives can be combined to build complex neural networks using human-understandable syntax.

## Building Your LLM

In this course, you will:
- Learn how JAX's XLA compiler works and how to use it
- Build and train a small LLM with 20 million parameters from scratch
- Define neural network architectures in JAX using Flax/NNX
- Learn ecosystem tools:
  - **Grain** — Data loading for large datasets
  - **Optax** — Optimizers and gradient processing
  - **Orbax** — Model checkpointing at scale

## Instructor

Chris Achard is Developer Relations Engineer on Google's TPU Software team, ensuring that Google's libraries and frameworks provide what developers need to build, train, and serve advanced AI models.

## The Same Approach as Google

The steps you take to build and train this mini GPT model are the same that Google uses to build more powerful LLMs like Gemini. This gives you hands-on experience with the core techniques behind modern AI model development.

## Key JAX Advantages

- **Functional programming paradigm** — Write composable, pure functions
- **Flexible compilation** — Automatic graph building via JIT without manual definition
- **Hardware agnostic** — Same code on CPUs, GPUs, TPUs
- **Research-friendly** — Support for advanced techniques like higher-order derivatives
- **Production-ready** — Scale to thousands of devices for large-scale training
