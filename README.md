# BERT Implementation from Scratch

A simplified implementation of BERT (Bidirectional Encoder Representations from Transformers) built from scratch using PyTorch, trained on WikiText-2 dataset with Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) objectives.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training Details](#training-details)
- [Results and Demonstration](#results-and-demonstration)
- [Implementation Details](#implementation-details)
- [References](#references)

## Overview

BERT (Bidirectional Encoder Representations from Transformers) is a revolutionary NLP model introduced by Devlin et al. in 2018. This implementation:

- **Built from scratch** - No pre-trained models, all layers implemented manually
- **Encoder-only Transformer** - Multi-head self-attention and feed-forward networks
- **Dual training objectives** - MLM (Masked Language Modeling) and NSP (Next Sentence Prediction)
- **Simplified architecture** - Smaller than BERT-base for faster training and learning
- **WikiText-2 dataset** - Small English corpus suitable for educational purposes

### Key Features

✓ Complete Transformer encoder with multi-head self-attention
✓ Three types of embeddings: token, position, and segment
✓ MLM with 80/10/10 masking strategy
✓ NSP for understanding sentence relationships
✓ Interactive demonstration of learned representations
✓ Comprehensive training metrics and visualization

## Architecture

### Model Configuration

This implementation uses a smaller architecture compared to BERT-base for faster training:

| Parameter | BERT-base | This Implementation |
|-----------|-----------|-------------------|
| Hidden Size | 768 | 256 |
| Attention Heads | 12 | 4 |
| Transformer Layers | 12 | 4 |
| Feed-Forward Size | 3072 | 1024 |
| Max Sequence Length | 512 | 128 |
| Vocabulary Size | 30,522 | 30,522 |
| Parameters | ~110M | ~8M |

### Components

1. **BERTEmbeddings**: Combines token, position, and segment embeddings
2. **MultiHeadSelfAttention**: Scaled dot-product attention mechanism
3. **BERTLayer**: Single transformer layer with attention and feed-forward network
4. **BERTEncoder**: Stack of transformer layers
5. **BERTPooler**: Pools [CLS] token for sentence-level tasks
6. **MLM Head**: Predicts masked tokens
7. **NSP Head**: Binary classification for sentence pairs

## Project Structure

```
NLP_PS/
├── config.py              # Configuration class with hyperparameters
├── bert_model.py          # BERT architecture implementation
├── dataset.py             # Data loading and preprocessing
├── train.py               # Training script
├── demo.py                # Interactive demonstration
├── visualize.py           # Training metrics visualization
├── README.md              # This file
└── venv/                  # Virtual environment
```

## Installation

### Prerequisites

- Python 3.7+
- Virtual environment (already set up in `venv/`)

### Setup

All dependencies are already installed in the virtual environment:

```bash
# Activate virtual environment (if needed)
# On Windows Git Bash:
source venv/Scripts/activate

# On Windows CMD:
venv\Scripts\activate.bat

# On Windows PowerShell:
venv\Scripts\Activate.ps1
```

**Installed packages:**
- PyTorch 2.9.0 (CPU version)
- Transformers 4.57.1
- Datasets 4.4.1
- NumPy, tqdm, and other dependencies

## Usage

### 1. Test Dataset Preparation

Before training, test the dataset preparation:

```bash
venv/Scripts/python dataset.py
```

This will:
- Download WikiText-2 dataset
- Create sentence pairs for NSP
- Apply MLM masking strategy
- Show sample batches

### 2. Train the Model

Start training with default configuration:

```bash
venv/Scripts/python train.py
```

**Training parameters** (configurable in `config.py`):
- Batch size: 16
- Learning rate: 1e-4
- Epochs: 3
- Max sequence length: 128
- MLM probability: 15%

**Expected output:**
```
================================================================================
BERT Pre-training from Scratch
================================================================================

Configuration: BERTConfig(hidden_size=256, num_layers=4)
Device: cpu

Preparing data...
Training batches: 482
Validation batches: 54

Initializing model...
Total parameters: 8,234,752
Trainable parameters: 8,234,752

Starting training...
================================================================================
Epoch 1/3
================================================================================
Epoch 1/3: 100%|████████| 482/482 [XX:XX<00:00,  X.XXit/s, loss=X.XX, mlm=X.XX, nsp=X.XX]
...
```

**Saved files:**
- `best_model.pt` - Best model checkpoint (lowest validation loss)
- `checkpoint_epoch_N.pt` - Checkpoint for each epoch
- `final_model.pt` - Final model after all epochs
- `training_history.json` - Metrics for all epochs

### 3. Visualize Training Metrics

After training, visualize the results:

```bash
venv/Scripts/python visualize.py
```

This generates:
- `training_metrics.png` - Plots of losses and accuracy
- Console output with training summary

### 4. Demonstrate Learned Representations

Run the interactive demonstration:

```bash
venv/Scripts/python demo.py
```

**Features:**
- **MLM Demo**: Shows top-k predictions for masked tokens
- **NSP Demo**: Predicts if two sentences are consecutive
- **Interactive Mode**: Test with your own inputs

**Example MLM output:**
```
Original text: The quick brown fox jumps over the lazy dog
Masked text: [CLS] The quick [MASK] fox jumps over the [MASK] dog [SEP]

Position 2 - Original token: 'brown'
Top 5 predictions:
  ✓ brown          (0.3421)
    red            (0.2156)
    black          (0.1543)
    white          (0.0987)
    big            (0.0654)
```

**Example NSP output:**
```
Sentence A: The weather is beautiful today.
Sentence B: We should go for a walk in the park.

Prediction: IS NEXT
Confidence: 0.8234
  - Probability IS NEXT: 0.8234
  - Probability NOT NEXT: 0.1766
```

## Training Details

### Masked Language Modeling (MLM)

**Objective**: Predict masked tokens in input sequence

**Masking Strategy** (as per BERT paper):
- Select 15% of tokens randomly for masking
- Of selected tokens:
  - 80% replaced with `[MASK]`
  - 10% replaced with random token
  - 10% kept unchanged

**Why this strategy?**
- Forces model to maintain representations for all tokens
- Prevents overfitting to `[MASK]` token
- Improves robustness

### Next Sentence Prediction (NSP)

**Objective**: Predict if sentence B follows sentence A

**Data Creation**:
- 50% positive examples: Consecutive sentences from corpus
- 50% negative examples: Random sentence pairs

**Purpose**:
- Learn relationships between sentences
- Useful for downstream tasks like QA, NLI

### Joint Training

Both losses are combined:
```
Total Loss = MLM Loss + NSP Loss
```

Both tasks are optimized simultaneously, following the BERT paper.

### Optimization

- **Optimizer**: AdamW with weight decay (0.01)
- **Learning Rate**: 1e-4 with linear warmup
- **Warmup Steps**: 1000
- **Gradient Clipping**: Max norm = 1.0
- **Dropout**: 0.1 for hidden states and attention

## Results and Demonstration

### Expected Performance

With the simplified architecture on WikiText-2:

- **MLM Loss**: Should decrease from ~8-10 to ~4-6
- **NSP Accuracy**: Should reach 60-75%
- **Training Time**: ~30-60 minutes per epoch on CPU (depending on hardware)

### Model Capabilities

After training, the model can:

1. **Predict masked words** with contextual understanding
2. **Determine sentence relationships** (consecutive vs. random)
3. **Generate meaningful representations** for downstream tasks

### Evaluation Metrics

Track these during training:
- Training/Validation MLM loss
- Training/Validation NSP loss
- NSP accuracy on validation set
- Total combined loss

## Implementation Details

### Key Architectural Choices

1. **Self-Attention Mechanism**
   - Scaled dot-product attention
   - Multiple attention heads for different representation subspaces
   - Residual connections and layer normalization

2. **Position Embeddings**
   - Learned absolute position embeddings
   - Max sequence length: 128 tokens

3. **Segment Embeddings**
   - Distinguish between sentence A (0) and sentence B (1)
   - Essential for NSP task

4. **Activation Functions**
   - GELU (Gaussian Error Linear Unit) as in BERT paper
   - More sophisticated than ReLU

5. **Special Tokens**
   - `[CLS]`: Classification token (position 0)
   - `[SEP]`: Separator between sentences
   - `[MASK]`: Mask token for MLM
   - `[PAD]`: Padding token

### Code Organization

**`config.py`**: Centralized configuration
- Model architecture parameters
- Training hyperparameters
- Special token IDs

**`bert_model.py`**: Complete model implementation
- All transformer components
- MLM and NSP heads
- No external transformer libraries used

**`dataset.py`**: Data preparation
- WikiText-2 loading
- Sentence pair creation
- MLM masking implementation
- PyTorch Dataset/DataLoader

**`train.py`**: Training loop
- Epoch-based training
- Validation evaluation
- Checkpoint saving
- Metrics tracking

**`demo.py`**: Interactive demonstration
- MLM predictions with top-k
- NSP predictions with confidence
- User input testing

## References

### Papers

1. **BERT Paper**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
   Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018)

2. **Transformer Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
   Vaswani, A., et al. (2017)

### Educational Resources

- [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
- [BERT Explained (Sebastian Ruder)](https://ruder.io/state-of-the-art-bert/)

### Dataset

- **WikiText-2**: [HuggingFace Datasets](https://huggingface.co/datasets/wikitext)
- Subset of Wikipedia articles
- ~2M training tokens, ~200K validation tokens

## Understanding BERT

### Why BERT Was Revolutionary

1. **Bidirectional Context**: Unlike GPT (left-to-right), BERT sees context from both directions
2. **Transfer Learning**: Pre-train once, fine-tune for many tasks
3. **State-of-the-Art**: Achieved SOTA on 11 NLP tasks upon release
4. **Simple and Effective**: Transformer encoder + two simple training tasks

### How BERT Learns

**Pre-training Phase** (this implementation):
- Learn general language representations
- No task-specific architecture
- Uses MLM and NSP objectives

**Fine-tuning Phase** (not in this implementation):
- Adapt to specific tasks (classification, QA, etc.)
- Add task-specific output layers
- Fine-tune all parameters

### What Makes This Educational

This implementation is designed for learning:
- ✓ Small enough to train on CPU
- ✓ All code visible and understandable
- ✓ Follows BERT paper closely
- ✓ Extensive comments and documentation
- ✓ Interactive demonstrations

## Troubleshooting

### Common Issues

**1. Out of Memory**
- Reduce `batch_size` in `config.py`
- Reduce `max_seq_length`
- Reduce `hidden_size` or `num_hidden_layers`

**2. Slow Training**
- This is expected on CPU
- Consider reducing `num_epochs` or `max_steps`
- Use GPU if available (set `config.device = 'cuda'`)

**3. Poor Performance**
- Train for more epochs
- Increase model size (hidden_size, layers)
- Check data quality

**4. Dataset Download Issues**
- Ensure internet connection
- HuggingFace datasets are auto-downloaded
- Check proxy settings if behind firewall

## Future Enhancements

Possible extensions to this implementation:

1. **Fine-tuning**: Add downstream task examples (classification, NER, QA)
2. **GPU Support**: Add CUDA optimization
3. **Larger Models**: Implement BERT-base or BERT-large configurations
4. **Advanced Features**: Implement whole word masking, dynamic masking
5. **Evaluation**: Add perplexity calculation, probing tasks
6. **Optimization**: Mixed precision training, gradient accumulation

## License

This is an educational implementation for learning purposes.

## Acknowledgments

- Original BERT paper by Devlin et al.
- HuggingFace for transformers library and datasets
- PyTorch team for the deep learning framework
- WikiText dataset creators

---

**Built for Inter-IIT Tech Meet Bootcamp**

For questions or issues, refer to the BERT paper and PyTorch documentation.
