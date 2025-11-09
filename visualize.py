"""
Visualize training metrics and model performance
"""

import json
import matplotlib.pyplot as plt
import os


def plot_training_history(history_file='training_history.json'):
    """
    Plot training and validation metrics from training history
    """
    if not os.path.exists(history_file):
        print(f"Training history file not found: {history_file}")
        return

    # Load training history
    with open(history_file, 'r') as f:
        history = json.load(f)

    epochs = [item['epoch'] for item in history]
    train_loss = [item['train_loss'] for item in history]
    val_loss = [item['val_loss'] for item in history]
    train_mlm_loss = [item['train_mlm_loss'] for item in history]
    val_mlm_loss = [item['val_mlm_loss'] for item in history]
    train_nsp_loss = [item['train_nsp_loss'] for item in history]
    val_nsp_loss = [item['val_nsp_loss'] for item in history]
    val_nsp_accuracy = [item['val_nsp_accuracy'] for item in history]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('BERT Training Metrics', fontsize=16, fontweight='bold')

    # Plot 1: Total Loss
    axes[0, 0].plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, 'r-o', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss (MLM + NSP)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: MLM Loss
    axes[0, 1].plot(epochs, train_mlm_loss, 'b-o', label='Train MLM Loss', linewidth=2)
    axes[0, 1].plot(epochs, val_mlm_loss, 'r-o', label='Val MLM Loss', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Masked Language Modeling Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: NSP Loss
    axes[1, 0].plot(epochs, train_nsp_loss, 'b-o', label='Train NSP Loss', linewidth=2)
    axes[1, 0].plot(epochs, val_nsp_loss, 'r-o', label='Val NSP Loss', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Next Sentence Prediction Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: NSP Accuracy
    axes[1, 1].plot(epochs, val_nsp_accuracy, 'g-o', label='Val NSP Accuracy', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Next Sentence Prediction Accuracy')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Training metrics plot saved to: training_metrics.png")
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"\nTotal epochs: {len(epochs)}")
    print(f"\nFinal metrics:")
    print(f"  Train Loss: {train_loss[-1]:.4f} (MLM: {train_mlm_loss[-1]:.4f}, NSP: {train_nsp_loss[-1]:.4f})")
    print(f"  Val Loss: {val_loss[-1]:.4f} (MLM: {val_mlm_loss[-1]:.4f}, NSP: {val_nsp_loss[-1]:.4f})")
    print(f"  Val NSP Accuracy: {val_nsp_accuracy[-1]:.4f}")

    print(f"\nBest metrics:")
    best_epoch = val_loss.index(min(val_loss))
    print(f"  Best epoch: {epochs[best_epoch]}")
    print(f"  Best Val Loss: {val_loss[best_epoch]:.4f}")
    print(f"  NSP Accuracy at best epoch: {val_nsp_accuracy[best_epoch]:.4f}")


if __name__ == "__main__":
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        plot_training_history()
    except ImportError:
        print("matplotlib not installed. Installing...")
        import subprocess
        subprocess.run(['venv/Scripts/pip', 'install', 'matplotlib'])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plot_training_history()
