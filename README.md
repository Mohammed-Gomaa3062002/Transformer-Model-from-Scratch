# Transformer Model from Scratch

This project implements a **Transformer model** from scratch using **PyTorch**, following the architecture described in the seminal paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). The Transformer model is designed for sequence-to-sequence tasks, such as machine translation, and leverages **self-attention mechanisms** to capture contextual relationships within input sequences.

## Features

- **Multi-Head Attention**: Implements scaled dot-product attention with multiple heads for parallel processing.
- **Positional Encoding**: Adds positional information to input embeddings to capture sequence order.
- **Encoder-Decoder Architecture**: Includes both encoder and decoder layers with residual connections and layer normalization.
- **Customizable Parameters**: Supports flexible configuration of model dimensions, number of layers, and attention heads.
- **Efficient Training**: Utilizes GPU acceleration (if available) for faster training.

## Technologies Used

- **PyTorch**: Primary deep learning framework for building and training the model.
- **NumPy**: For numerical computations and matrix operations.
- **CUDA**: For GPU acceleration (optional).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Mohamed-Gomaaa3062002/Transformer-from-Scratch.git
   cd Transformer-from-Scratch
   ```

2. Install dependencies:
   ```bash
   pip install torch numpy
   ```

3. Ensure CUDA is installed (optional for GPU acceleration):
   - Install CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).
   - Verify installation:
     ```bash
     nvcc --version
     ```

## Usage

1. **Model Initialization**:
   - Define the model parameters (e.g., `d_model`, `num_heads`, `num_layers`).
   - Initialize the Transformer model:
     ```python
     from transformer import Transformer

     model = Transformer(
         d_model=512,
         ffn_hidden=2048,
         num_heads=8,
         drop_prob=0.1,
         num_layers=6,
         max_sequence_length=100,
         kn_vocab_size=10000,
         english_to_index=english_to_index,
         kannada_to_index=kannada_to_index,
         START_TOKEN="<SOS>",
         END_TOKEN="<EOS>",
         PADDING_TOKEN="<PAD>"
     )
     ```

2. **Training**:
   - Prepare your dataset (e.g., tokenized sentences for machine translation).
   - Train the model using a suitable loss function (e.g., CrossEntropyLoss) and optimizer (e.g., Adam).

3. **Inference**:
   - Use the trained model to generate predictions for new input sequences.

## Example

```python
# Example input sequences (batch of sentences)
x = ["This is a sample sentence.", "Another example sentence."]
y = ["Translation of the sentence.", "Another translation."]

# Forward pass
output = model(x, y)
print(output)
```

## Project Structure

```
Transformer-from-Scratch/
├── transformer.py          # Transformer model implementation
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

**Author**: Mohamed Gomaa 
**Contact**: mogommaa2002@gmail.com 

---
