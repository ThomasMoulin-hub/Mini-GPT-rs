# Mini-GPT-rs

A minimal, educational implementation of a GPT-style decoder-only Transformer in Rust, built from scratch using the `tch-rs` (Rust bindings for PyTorch).

This project is intended for learning purposes, demonstrating the core components of a Transformer architecture in a clear and concise Rust codebase.

## Features

- **Decoder-Only Architecture:** A simple GPT-style model.
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
cd mini-gpt-rs
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
$env:LIBTORCH="<full-path-to-your-project>\mini-gpt\.venv\Lib\site-packages	orch"

# Example:
# $env:LIBTORCH="C:\Users\YourUser\projects\mini-gpt\.venv\Lib\site-packages	orch"
```
> **Note:** For a permanent setup, you can add this variable to your system's or user's environment variables through the System Properties dialog.


## Running the Project

Once the `LIBTORCH` environment variable is set correctly, you can build and run the project using Cargo.

```bash
cargo run
```

You should see the training process start, printing the loss every 10 epochs:

```
Starting mini-gpt training...
Starting training for 50 epochs...
Epoch 10 Loss: 10.0712
Epoch 20 Loss: 9.9419
...
Training finished!
mini-gpt training finished.
```

## Project Structure

- `src/main.rs`: Main entry point, sets up the training configuration and starts the training loop.
- `src/model.rs`: Defines the main `Transformer` struct and its overall architecture.
- `src/layers.rs`: Implements the core Transformer layers (`PositionalEncoding`, `MultiHeadAttention`, `FeedForward`, `TransformerBlock`).
- `src/training.rs`: Contains the `train` function with the main training loop (data generation, forward pass, loss calculation, backpropagation).
- `src/data.rs`: Includes simple data generation functions for demonstration purposes.
