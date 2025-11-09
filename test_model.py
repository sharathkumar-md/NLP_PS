"""
Quick test script to verify model implementation
Tests model forward pass without training
"""

import torch
from config import BERTConfig
from bert_model import BERTForPreTraining


def test_model_forward_pass():
    """
    Test that the model can perform a forward pass
    """
    print("=" * 80)
    print("BERT MODEL IMPLEMENTATION TEST")
    print("=" * 80)

    # Create config
    config = BERTConfig()
    print(f"\n1. Configuration: {config}")

    # Initialize model
    print("\n2. Initializing model...")
    model = BERTForPreTraining(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Create dummy input
    batch_size = 4
    seq_length = 32

    print(f"\n3. Creating dummy input (batch_size={batch_size}, seq_length={seq_length})...")
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    masked_lm_labels = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    next_sentence_label = torch.randint(0, 2, (batch_size,))

    # Test forward pass without labels
    print("\n4. Testing forward pass (without labels)...")
    with torch.no_grad():
        mlm_logits, nsp_logits = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

    print(f"   MLM logits shape: {mlm_logits.shape} (expected: [{batch_size}, {seq_length}, {config.vocab_size}])")
    print(f"   NSP logits shape: {nsp_logits.shape} (expected: [{batch_size}, 2])")

    assert mlm_logits.shape == (batch_size, seq_length, config.vocab_size), "MLM logits shape mismatch!"
    assert nsp_logits.shape == (batch_size, 2), "NSP logits shape mismatch!"
    print("   [OK] Shapes correct!")

    # Test forward pass with labels
    print("\n5. Testing forward pass (with labels)...")
    with torch.no_grad():
        total_loss, mlm_loss, nsp_loss = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            masked_lm_labels=masked_lm_labels,
            next_sentence_label=next_sentence_label
        )

    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   MLM loss: {mlm_loss.item():.4f}")
    print(f"   NSP loss: {nsp_loss.item():.4f}")

    assert total_loss.item() > 0, "Loss should be positive!"
    print("   [OK] Loss computation works!")

    # Test embeddings
    print("\n6. Testing embeddings...")
    embeddings = model.bert.embeddings(input_ids, token_type_ids)
    print(f"   Embeddings shape: {embeddings.shape} (expected: [{batch_size}, {seq_length}, {config.hidden_size}])")
    assert embeddings.shape == (batch_size, seq_length, config.hidden_size), "Embeddings shape mismatch!"
    print("   [OK] Embeddings work!")

    # Test attention mechanism
    print("\n7. Testing attention mechanism...")
    attention_layer = model.bert.encoder.layers[0].attention
    attention_output = attention_layer(embeddings)
    print(f"   Attention output shape: {attention_output.shape}")
    assert attention_output.shape == embeddings.shape, "Attention output shape mismatch!"
    print("   [OK] Attention works!")

    # Test pooler
    print("\n8. Testing pooler...")
    sequence_output, pooled_output = model.bert(input_ids, token_type_ids, attention_mask)
    print(f"   Sequence output shape: {sequence_output.shape} (expected: [{batch_size}, {seq_length}, {config.hidden_size}])")
    print(f"   Pooled output shape: {pooled_output.shape} (expected: [{batch_size}, {config.hidden_size}])")
    assert pooled_output.shape == (batch_size, config.hidden_size), "Pooled output shape mismatch!"
    print("   [OK] Pooler works!")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe BERT model implementation is working correctly!")
    print("You can now proceed to training with: python train.py")
    print("=" * 80)


if __name__ == "__main__":
    test_model_forward_pass()
