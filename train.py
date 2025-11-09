"""
Training script for BERT pre-training
Implements joint training on MLM and NSP tasks
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import os
import json

from config import BERTConfig
from bert_model import BERTForPreTraining
from dataset import create_dataloaders


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, train_loader, optimizer, scheduler, config, epoch):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    total_mlm_loss = 0
    total_nsp_loss = 0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")

    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        input_ids = batch['input_ids'].to(config.device)
        token_type_ids = batch['token_type_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        masked_lm_labels = batch['masked_lm_labels'].to(config.device)
        next_sentence_label = batch['next_sentence_label'].to(config.device)

        # Forward pass
        optimizer.zero_grad()
        loss, mlm_loss, nsp_loss = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            masked_lm_labels=masked_lm_labels,
            next_sentence_label=next_sentence_label
        )

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()

        # Track metrics
        total_loss += loss.item()
        total_mlm_loss += mlm_loss.item()
        total_nsp_loss += nsp_loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / num_batches,
            'mlm': total_mlm_loss / num_batches,
            'nsp': total_nsp_loss / num_batches,
            'lr': scheduler.get_last_lr()[0]
        })

    return total_loss / num_batches, total_mlm_loss / num_batches, total_nsp_loss / num_batches


def evaluate(model, val_loader, config):
    """
    Evaluate on validation set
    """
    model.eval()
    total_loss = 0
    total_mlm_loss = 0
    total_nsp_loss = 0
    total_nsp_correct = 0
    num_batches = 0
    num_examples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move to device
            input_ids = batch['input_ids'].to(config.device)
            token_type_ids = batch['token_type_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            masked_lm_labels = batch['masked_lm_labels'].to(config.device)
            next_sentence_label = batch['next_sentence_label'].to(config.device)

            # Forward pass
            loss, mlm_loss, nsp_loss = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                masked_lm_labels=masked_lm_labels,
                next_sentence_label=next_sentence_label
            )

            # Get NSP predictions
            _, nsp_logits = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
            nsp_predictions = torch.argmax(nsp_logits, dim=-1)
            total_nsp_correct += (nsp_predictions == next_sentence_label).sum().item()

            # Track metrics
            total_loss += loss.item()
            total_mlm_loss += mlm_loss.item()
            total_nsp_loss += nsp_loss.item()
            num_batches += 1
            num_examples += input_ids.size(0)

    avg_loss = total_loss / num_batches
    avg_mlm_loss = total_mlm_loss / num_batches
    avg_nsp_loss = total_nsp_loss / num_batches
    nsp_accuracy = total_nsp_correct / num_examples

    return avg_loss, avg_mlm_loss, avg_nsp_loss, nsp_accuracy


def save_checkpoint(model, optimizer, scheduler, epoch, config, metrics, filename='checkpoint.pt'):
    """
    Save model checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': vars(config),
        'metrics': metrics
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, scheduler, config, checkpoint_path='best_model.pt'):
    """
    Load model checkpoint if it exists and matches architecture
    Returns: (start_epoch, best_val_loss, train_history, load_optimizer) or (0, inf, [], False) if not resuming
    """
    if not os.path.exists(checkpoint_path):
        print(f"\nNo checkpoint found at {checkpoint_path}")
        print("Starting training from scratch...")
        return 0, float('inf'), [], False

    try:
        print(f"\nAttempting to load checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)

        # Check if architecture matches
        saved_config = checkpoint.get('config', {})
        config_matches = (
            saved_config.get('hidden_size') == config.hidden_size and
            saved_config.get('num_hidden_layers') == config.num_hidden_layers and
            saved_config.get('num_attention_heads') == config.num_attention_heads
        )

        if not config_matches:
            print("\n" + "=" * 80)
            print("WARNING: Checkpoint architecture doesn't match current config!")
            print("=" * 80)
            print(f"Checkpoint: hidden_size={saved_config.get('hidden_size')}, "
                  f"layers={saved_config.get('num_hidden_layers')}, "
                  f"heads={saved_config.get('num_attention_heads')}")
            print(f"Current:    hidden_size={config.hidden_size}, "
                  f"layers={config.num_hidden_layers}, "
                  f"heads={config.num_attention_heads}")
            print("\nStarting training from scratch with NEW architecture...")
            print("=" * 80)
            return 0, float('inf'), [], False

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])

        # Check if training parameters changed
        training_params_changed = (
            saved_config.get('batch_size') != config.batch_size or
            saved_config.get('max_seq_length') != config.max_seq_length or
            saved_config.get('learning_rate') != config.learning_rate
        )

        load_optimizer = not training_params_changed

        if training_params_changed:
            print("\n" + "=" * 80)
            print("NOTE: Training parameters changed - will reset optimizer/scheduler")
            print("=" * 80)
            print(f"Checkpoint: batch_size={saved_config.get('batch_size')}, "
                  f"seq_length={saved_config.get('max_seq_length')}, "
                  f"lr={saved_config.get('learning_rate')}")
            print(f"Current:    batch_size={config.batch_size}, "
                  f"seq_length={config.max_seq_length}, "
                  f"lr={config.learning_rate}")
            print("\nModel weights loaded, but optimizer/scheduler will be freshly initialized")
            print("=" * 80)
        else:
            # Load optimizer and scheduler only if parameters unchanged
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        metrics = checkpoint.get('metrics', {})
        best_val_loss = metrics.get('val_loss', float('inf'))

        print(f"\n✓ Checkpoint loaded successfully!")
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Best validation loss so far: {best_val_loss:.4f}")

        # Try to load training history
        train_history = []
        if os.path.exists('training_history.json'):
            with open('training_history.json', 'r') as f:
                train_history = json.load(f)
            print(f"  Loaded {len(train_history)} epochs of training history")

        return start_epoch, best_val_loss, train_history, load_optimizer

    except Exception as e:
        print(f"\nError loading checkpoint: {e}")
        print("Starting training from scratch...")
        return 0, float('inf'), [], False


def main():
    """
    Main training function
    """
    print("=" * 80)
    print("BERT Pre-training from Scratch")
    print("=" * 80)

    # Configuration
    config = BERTConfig()
    print(f"\nConfiguration: {config}")
    print(f"Device: {config.device}")

    # Create dataloaders
    print("\nPreparing data...")
    train_loader, val_loader, tokenizer = create_dataloaders(config)
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Initialize model
    print("\nInitializing model...")
    model = BERTForPreTraining(config)
    model.to(config.device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer and scheduler (initialize first)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)

    # Try to load checkpoint and resume training
    start_epoch, best_val_loss, train_history, loaded_optimizer = load_checkpoint(model, optimizer, scheduler, config)

    # Re-initialize optimizer/scheduler if training params changed
    if start_epoch > 0 and not loaded_optimizer:
        print("\nRe-initializing optimizer and scheduler with new parameters...")
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        # Calculate remaining steps for scheduler
        remaining_steps = len(train_loader) * (config.num_epochs - start_epoch)
        scheduler = get_linear_schedule_with_warmup(optimizer, config.warmup_steps, remaining_steps)

    # Training loop
    if start_epoch == 0:
        print("\nStarting training from scratch...")
    else:
        print(f"\nContinuing training from epoch {start_epoch + 1}...")
        print(f"Will train from epoch {start_epoch + 1} to epoch {config.num_epochs}")

    for epoch in range(start_epoch, config.num_epochs):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'=' * 80}")

        # Train
        train_loss, train_mlm_loss, train_nsp_loss = train_epoch(
            model, train_loader, optimizer, scheduler, config, epoch
        )

        # Evaluate
        print("\nEvaluating...")
        val_loss, val_mlm_loss, val_nsp_loss, val_nsp_accuracy = evaluate(model, val_loader, config)

        # Print metrics
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f} (MLM: {train_mlm_loss:.4f}, NSP: {train_nsp_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (MLM: {val_mlm_loss:.4f}, NSP: {val_nsp_loss:.4f})")
        print(f"  Val NSP Accuracy: {val_nsp_accuracy:.4f}")

        # Save metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_mlm_loss': train_mlm_loss,
            'train_nsp_loss': train_nsp_loss,
            'val_loss': val_loss,
            'val_mlm_loss': val_mlm_loss,
            'val_nsp_loss': val_nsp_loss,
            'val_nsp_accuracy': val_nsp_accuracy
        }
        train_history.append(epoch_metrics)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, config, epoch_metrics, 'best_model.pt')

        # Save checkpoint every epoch
        save_checkpoint(model, optimizer, scheduler, epoch, config, epoch_metrics, f'checkpoint_epoch_{epoch + 1}.pt')

    # Save final model
    save_checkpoint(model, optimizer, scheduler, config.num_epochs - 1, config, train_history[-1], 'final_model.pt')

    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(train_history, f, indent=2)

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Model saved to: best_model.pt")


if __name__ == "__main__":
    main()
