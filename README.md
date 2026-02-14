# Mini-GPT-rs

A minimal, educational implementation of a GPT-style decoder-only Transformer in Rust, built from scratch using the `tch-rs` (Rust bindings for PyTorch).

This project is intended for learning purposes, demonstrating the core components of a Transformer architecture in a clear and concise Rust codebase, including a data pipeline that tokenizes the "TinyShakespeare" dataset for training.

## Features

- **Decoder-Only Architecture:** A simple GPT-style model.
- **Data Pipeline:**
  - Automatically downloads the "TinyShakespeare" dataset.
  - Trains a **Byte-Pair Encoding (BPE)** tokenizer from scratch on the dataset.
  - Caches the trained tokenizer in `data/tokenizer.json`.
  - Generates batches of real text data for training.
- **Core Transformer Components:**
  - Scaled Dot-Product Multi-Head Attention
  - Positional Encoding
  - Position-wise Feed-Forward Networks
  - Layer Normalization and Residual Connections
- **Pure Rust Implementation:** The model logic is written entirely in Rust.
- **PyTorch Backend:** Leverages the power and efficiency of the PyTorch C++ library (LibTorch) via `tch-rs` for tensor operations.

## Prerequisites

This project has a specific dependency setup due to `tch-rs` requiring a PyTorch C++ library (LibTorch) to be present on your system. The recommended way to handle this is by using a Python environment.

1.  **Rust:** You need the Rust toolchain installed. If you don't have it, get it from [rustup.rs](https://rustup.rs/).
2.  **Python:** A recent version of Python (e.g., 3.8+) is required.
3.  **PyTorch:** We will install PyTorch in a Python virtual environment, which will also download the required C++ library files.

## Setup and Installation

Follow these steps carefully to configure your environment.

### 1. Clone the Repository

```bash
git clone https://github.com/ThomasMoulin-hub/Mini-GPT-rs.git
cd Mini-GPT-rs
```

### 2. Set up the Python Virtual Environment

It's crucial to use a virtual environment to avoid conflicts with system-wide Python packages.

```bash
# Create a virtual environment named '.venv'
python -m venv .venv

# Activate the environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On macOS/Linux:
# source .venv/bin/activate
```

### 3. Install PyTorch

With the virtual environment active, install `torch`. This provides the underlying C++ libraries needed by `tch-rs`.

```bash
pip install torch torchvision torchaudio
```

### 4. Set the `LIBTORCH` Environment Variable

The `tch-rs` crate finds its dependencies using an environment variable named `LIBTORCH`. You must set this variable to point to the `torch` installation directory within your virtual environment.

**The path will look something like this:** `<project-directory>\.venv\Lib\site-packages	orch`

**To set the variable (for the current session):**

```powershell
# On Windows (PowerShell)
# Make sure to replace <full-path-to-your-project> with the actual absolute path
$env:LIBTORCH="<full-path-to-your-project>\mini-gpt\.venv\Lib\site-packages\torch"

# Example:
# $env:LIBTORCH="C:\Users\YourUser\projects\mini-gpt\.venv\Lib\site-packages\torch"
```
> **Note:** For a permanent setup, you can add this variable to your system's or user's environment variables through the System Properties dialog.


## Running the Project

Once the `LIBTORCH` environment variable is set correctly, you can build and run the project using Cargo.

### Training and Generation

```bash
cargo run
```

The first time you run it, the application will:
1. Download the TinyShakespeare dataset
2. Train a new BPE tokenizer (saved to `data/tokenizer.json`)
3. Train the model for 50 epochs
4. Save the trained model to `data/model.pt`
5. Enter interactive text generation mode

You should see the training process start, printing the loss every 10 epochs:

```
=== Mini-GPT Rust ===

Starting mini-gpt training...
Downloading TinyShakespeare dataset...
Dataset downloaded successfully.
Training a new tokenizer...
Tokenizer trained and saved.
Starting training for 50 epochs...
Epoch 10 Loss: 7.2134
Epoch 20 Loss: 6.8432
...
Saving model to data/model.pt...
Model saved successfully!
Training finished!

=== Text Generation Mode ===
Loading model from data/model.pt...
Model loaded successfully!

Enter your prompt (or 'quit' to exit):
Commands:
  quit - Exit the program
  clear - Clear screen

> To be or not to be
Generating 100 tokens...
..........
Generated text:
To be or not to be that is the question...
```

### Skip Training (Use Existing Model)

If you've already trained a model and just want to generate text:

```bash
cargo run -- --skip-training
```

This will load the existing model from `data/model.pt` and go straight to the interactive generation mode.

### Interactive Commands

- Type any text prompt to generate a continuation
- `quit` or `exit` - Exit the program
- `clear` - Clear the screen

## Project Structure

- `src/main.rs`: Main entry point. Orchestrates training and interactive text generation.
- `src/model.rs`: Defines the main `Transformer` struct and its overall architecture.
- `src/layers.rs`: Implements the core Transformer layers (`PositionalEncoding`, `MultiHeadAttention`, `FeedForward`, `TransformerBlock`).
- `src/training.rs`: Contains the `train` function with the main training loop and **causal masking** for decoder-only architecture.
- `src/generation.rs`: Implements text generation with temperature sampling and model loading utilities.
- `src/data.rs`: Implements the data pipeline. Includes functions to download the dataset, train/load a BPE tokenizer, and generate batches of tokenized text.
- `data/`: This directory is created automatically. It stores the `tinyshakespeare.txt` dataset, the trained `tokenizer.json`, and the saved model `model.pt`.

