# Poem Generation Project

## Overview
This project implements a poem generation model using a custom Generative Pre-trained Transformer (GPT) architecture, meticulously built from scratch to ensure full control over the model's design and functionality. The model is trained on a corpus of poems extracted from text files within a zip archive, processes the text data, and generates novel poetic sequences based on user-provided prompts. The implementation leverages Python, PyTorch for model training, and TensorFlow for text preprocessing, delivering a robust solution for creative text generation.

## Project Structure
The project consists of four main files:
- **Data_processing (1).py**: Handles data loading, preprocessing, and tokenization of the poem corpus.
- **Model (1) (1).py**: Defines the custom GPT model architecture, including self-attention, multi-head attention, feed-forward layers, and transformer blocks, all developed from the ground up.
- **Pretrain (1).py**: Manages the pretraining process of the GPT model on the processed poem corpus.
- **Poem_Generation_1.ipynb**: A Jupyter notebook for generating poems using the pretrained model and user prompts.

## Prerequisites
To run this project, ensure you have the following installed:
- Python 3.8 or higher
- PyTorch
- TensorFlow
- NumPy
- A zip file (`forms.zip`) containing `.txt` files with poems for training
- Optional: CUDA-enabled GPU for faster training (CPU also supported)

Install dependencies using:
```
pip install torch tensorflow numpy
```

## Setup
1. **Prepare the Data**:
   - Place the `forms.zip` file containing poem text files in the project directory.
   - Ensure the zip file contains `.txt` files with poem lines separated by newlines.

2. **Directory Structure**:
   ```
   project_directory/
   ├── forms.zip
   ├── Data_processing (1).py
   ├── Model (1) (1).py
   ├── Pretrain (1).py
   ├── Poem_Generation_1.ipynb
   ├── Poems.txt (generated during preprocessing)
   ├── pretrained_model.pth (generated after training)
   ├── tokenizer.pkl (generated during preprocessing)
   ```

## Usage
### 1. Data Preprocessing
Run `Data_processing (1).py` to:
- Extract poems from `forms.zip`.
- Save the corpus to `Poems.txt`.
- Tokenize the text and create n-gram sequences for training.
- Generate a `tokenizer.pkl` file for later use in poem generation.

```
python "Data_processing (1).py"
```

### 2. Model Training
Run `Pretrain (1).py` to train the GPT model:
- The model, crafted from scratch, uses a vocabulary size of 77,610, an embedding dimension of 256, a sequence length of 200, 8 attention heads, and a dropout rate of 0.1.
- Training runs for 1 epoch by default (adjustable in the script).
- The trained model is saved as `pretrained_model.pth`.

```
python "Pretrain (1).py"
```

### 3. Poem Generation
Open `Poem_Generation_1.ipynb` in Jupyter Notebook to generate poems:
- Load the pretrained model and tokenizer.
- Use the `generate_text` function with a prompt (e.g., "Once upon a time" or "I love water").
- The function generates text up to a maximum length of 200 tokens, with temperature sampling (default: 1.7) and top-k filtering (default: k=10) for randomness.

### Example Outputs
- **Prompt**: "Once upon a time"
  - **Output**: A sequence of words forming a poetic continuation, though it may include repetitive patterns due to limited training (e.g., "once upon a time for to for that to a but to and in i the and...").
- **Prompt**: "I love water"
  - **Output**: A similar poetic continuation (e.g., "i love water but but and a but the for the that...").

Note: The generated text may lack coherence due to limited training data or epochs. Increasing training epochs or fine-tuning hyperparameters can improve results.

## Model Architecture
The GPT model, developed entirely from scratch, consists of:
- **Embedding Layers**: Word and positional embeddings (256 dimensions).
- **Transformer Blocks**: 12 blocks, each with:
  - Multi-head attention (8 heads).
  - Feed-forward layers with GELU activation.
  - Layer normalization and dropout (0.1).
- **Output Layer**: Maps to the vocabulary size (77,610).

The model incorporates a causal mask to ensure autoregressive generation, where each token only attends to previous tokens.

## Data Processing Details
- **Input**: Poems from `forms.zip` are read and split into lines.
- **Preprocessing**: Lines are converted to lowercase, punctuation is removed, and text is tokenized using TensorFlow's `Tokenizer`.
- **N-grams**: Variable-length n-grams are created and padded to a fixed length of 200 tokens.
- **Dataset**: Converted to PyTorch tensors and loaded into a `DataLoader` with a batch size of 32.

## Training Details
- **Optimizer**: AdamW with a learning rate of 5e-4 and weight decay of 1e-2.
- **Loss Function**: Cross-entropy loss.
- **Training**: Single epoch (configurable), with loss printed per epoch.
- **Hardware**: Supports GPU (CUDA) or CPU.

## Limitations
- **Training Duration**: The model is trained for only one epoch, which may result in suboptimal poem quality.
- **Output Coherence**: Generated poems may be repetitive or lack poetic structure due to limited training data or hyperparameter tuning.
- **Tokenizer Dependency**: The tokenizer must be trained on the same dataset used for generation to avoid empty sequence errors.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.