# Quick Start Guide

## Fast Track to Running BERT

### Step 1: Verify Installation

Test that everything is set up correctly:

```bash
venv/Scripts/python test_model.py
```

**Expected output:**
```
ALL TESTS PASSED!
The BERT model implementation is working correctly!
```

### Step 2: Test Dataset Preparation (Optional)

See how data is preprocessed:

```bash
venv/Scripts/python dataset.py
```

This will download WikiText-2 and show sample batches.

### Step 3: Train the Model

Start training (takes ~30-60 minutes per epoch on CPU):

```bash
venv/Scripts/python train.py
```

**What happens:**
- Downloads WikiText-2 dataset automatically
- Trains for 3 epochs by default
- Saves checkpoints after each epoch
- Saves best model to `best_model.pt`

**Monitor progress:**
```
Epoch 1/3: 100%|████████| 482/482 [XX:XX<00:00, loss=X.XX, mlm=X.XX, nsp=X.XX]
Evaluating: 100%|████████| 54/54 [XX:XX<00:00]

Epoch 1 Results:
  Train Loss: X.XXXX (MLM: X.XXXX, NSP: X.XXXX)
  Val Loss: X.XXXX (MLM: X.XXXX, NSP: X.XXXX)
  Val NSP Accuracy: X.XXXX
```

### Step 4: Visualize Results

After training, visualize metrics:

```bash
venv/Scripts/python visualize.py
```

Generates `training_metrics.png` with loss and accuracy plots.

### Step 5: Test Your Model!

Run the interactive demonstration:

```bash
venv/Scripts/python demo.py
```

**Features:**
- See masked token predictions
- Test next sentence prediction
- Interactive mode for custom inputs

## Example Usage

### Masked Language Modeling

```
Original text: The quick brown fox jumps over the lazy dog
Masked text: The quick [MASK] fox jumps over the [MASK] dog

Position 2 - Original token: 'brown'
Top 5 predictions:
  [✓] brown (0.3421)
      red (0.2156)
      ...
```

### Next Sentence Prediction

```
Sentence A: The weather is beautiful today.
Sentence B: We should go for a walk in the park.

Prediction: IS NEXT
Confidence: 0.8234
```

## Adjusting Training Parameters

Edit `config.py` to customize:

```python
# Model size
self.hidden_size = 256  # Increase for larger model
self.num_hidden_layers = 4  # More layers = deeper model

# Training
self.batch_size = 16  # Reduce if out of memory
self.num_epochs = 3  # More epochs = longer training
self.learning_rate = 1e-4  # Adjust learning speed
```

## Troubleshooting

**Out of memory?**
- Reduce `batch_size` to 8 or 4
- Reduce `max_seq_length` to 64

**Training too slow?**
- Reduce `num_epochs` to 1 or 2
- Reduce `num_hidden_layers` to 2 or 3
- This is expected on CPU (GPU would be 10-100x faster)

**Want better results?**
- Train for more epochs (5-10)
- Increase model size (hidden_size=512, layers=6)
- Note: Larger models need more data and time

## File Outputs

After training, you'll have:

- `best_model.pt` - Best model checkpoint (use this for demo)
- `final_model.pt` - Model after last epoch
- `checkpoint_epoch_N.pt` - Checkpoint for each epoch
- `training_history.json` - All training metrics
- `training_metrics.png` - Visualization (after running visualize.py)

## Next Steps

1. **Understand the code**: Read through the implementation files
2. **Experiment**: Modify hyperparameters, try different texts
3. **Extend**: Add fine-tuning for downstream tasks
4. **Learn**: Compare with the BERT paper

## Key Files

- `config.py` - All hyperparameters
- `bert_model.py` - Model architecture (~350 lines)
- `dataset.py` - Data preprocessing
- `train.py` - Training loop
- `demo.py` - Interactive demonstration
- `test_model.py` - Quick tests

## Commands Cheat Sheet

```bash
# Test model
venv/Scripts/python test_model.py

# Test dataset
venv/Scripts/python dataset.py

# Train
venv/Scripts/python train.py

# Visualize
venv/Scripts/python visualize.py

# Demo
venv/Scripts/python demo.py
```

## Expected Timeline

- **Setup & Testing**: 5 minutes
- **Understanding Code**: 30-60 minutes
- **Training (3 epochs)**: 1.5-3 hours on CPU
- **Experimentation**: As long as you want!

---

**Ready to start? Run:**
```bash
venv/Scripts/python test_model.py && venv/Scripts/python train.py
```

Good luck! 🚀
