"""
Demonstration script for BERT model
Shows masked token predictions and Next Sentence Prediction capabilities
"""

import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import random

from config import BERTConfig
from bert_model import BERTForPreTraining


def load_model(checkpoint_path='best_model.pt'):
    """
    Load trained BERT model from checkpoint
    """
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    config = BERTConfig()
    model = BERTForPreTraining(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully (trained for {checkpoint['epoch'] + 1} epochs)")
    return model, config


def predict_masked_tokens(model, tokenizer, text, config, num_masks=3):
    """
    Demonstrate MLM by masking random words and predicting them

    Args:
        model: Trained BERT model
        tokenizer: BERT tokenizer
        text: Input text string
        config: BERT config
        num_masks: Number of tokens to mask
    """
    print("\n" + "=" * 80)
    print("MASKED LANGUAGE MODEL DEMONSTRATION")
    print("=" * 80)

    # Tokenize
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    if len(token_ids) == 0:
        print("Text is too short!")
        return

    # Select random positions to mask (avoid subwords starting with ##)
    maskable_positions = [i for i, token in enumerate(tokens) if not token.startswith('##')]
    if len(maskable_positions) < num_masks:
        num_masks = len(maskable_positions)

    masked_positions = random.sample(maskable_positions, num_masks)
    masked_positions.sort()

    # Create masked input
    masked_token_ids = token_ids.copy()
    original_tokens = []
    for pos in masked_positions:
        original_tokens.append(tokens[pos])
        masked_token_ids[pos] = config.mask_token_id

    # Add [CLS] and [SEP]
    input_ids = [config.cls_token_id] + masked_token_ids + [config.sep_token_id]
    attention_mask = [1] * len(input_ids)
    token_type_ids = [0] * len(input_ids)

    # Convert to tensors
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)

    # Get predictions
    with torch.no_grad():
        mlm_logits, _ = model(input_ids, token_type_ids, attention_mask)

    # Show results
    print(f"\nOriginal text: {text}")
    print(f"\nMasked text: {tokenizer.decode(input_ids[0])}")
    print("\nPredictions:")
    print("-" * 80)

    for i, pos in enumerate(masked_positions):
        actual_pos = pos + 1  # Account for [CLS] token
        logits = mlm_logits[0, actual_pos]
        probs = F.softmax(logits, dim=-1)

        # Get top 5 predictions
        top_k = 5
        top_probs, top_indices = torch.topk(probs, top_k)

        print(f"\nPosition {pos} - Original token: '{original_tokens[i]}'")
        print(f"Top {top_k} predictions:")
        for j in range(top_k):
            predicted_token = tokenizer.decode([top_indices[j].item()])
            probability = top_probs[j].item()
            marker = "✓" if predicted_token.strip() == original_tokens[i] else " "
            print(f"  {marker} {predicted_token:15s} ({probability:.4f})")


def predict_next_sentence(model, tokenizer, sentence_a, sentence_b, config):
    """
    Demonstrate NSP by predicting if sentence_b follows sentence_a

    Args:
        model: Trained BERT model
        tokenizer: BERT tokenizer
        sentence_a: First sentence
        sentence_b: Second sentence
        config: BERT config
    """
    # Tokenize
    tokens_a = tokenizer.encode(sentence_a, add_special_tokens=False)
    tokens_b = tokenizer.encode(sentence_b, add_special_tokens=False)

    # Create BERT input: [CLS] + tokens_a + [SEP] + tokens_b + [SEP]
    input_ids = [config.cls_token_id] + tokens_a + [config.sep_token_id] + tokens_b + [config.sep_token_id]
    token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
    attention_mask = [1] * len(input_ids)

    # Convert to tensors
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long)

    # Get predictions
    with torch.no_grad():
        _, nsp_logits = model(input_ids, token_type_ids, attention_mask)
        probs = F.softmax(nsp_logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()

    # Show results
    is_next_prob = probs[0, 1].item()
    not_next_prob = probs[0, 0].item()

    print(f"\nSentence A: {sentence_a}")
    print(f"Sentence B: {sentence_b}")
    print(f"\nPrediction: {'IS NEXT' if prediction == 1 else 'NOT NEXT'}")
    print(f"Confidence: {max(is_next_prob, not_next_prob):.4f}")
    print(f"  - Probability IS NEXT: {is_next_prob:.4f}")
    print(f"  - Probability NOT NEXT: {not_next_prob:.4f}")


def main():
    """
    Main demonstration function
    """
    print("=" * 80)
    print("BERT MODEL DEMONSTRATION")
    print("=" * 80)

    # Load model
    try:
        model, config = load_model('best_model.pt')
    except FileNotFoundError:
        print("\nModel checkpoint not found!")
        print("Please train the model first by running: python train.py")
        return

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # === MLM Demonstrations ===
    print("\n\n" + "=" * 80)
    print("PART 1: MASKED LANGUAGE MODELING (MLM)")
    print("=" * 80)

    # Example 1
    predict_masked_tokens(
        model, tokenizer,
        "The quick brown fox jumps over the lazy dog",
        config, num_masks=3
    )

    # Example 2
    predict_masked_tokens(
        model, tokenizer,
        "Machine learning is a subset of artificial intelligence",
        config, num_masks=3
    )

    # Example 3
    predict_masked_tokens(
        model, tokenizer,
        "Python is a popular programming language for data science",
        config, num_masks=3
    )

    # === NSP Demonstrations ===
    print("\n\n" + "=" * 80)
    print("PART 2: NEXT SENTENCE PREDICTION (NSP)")
    print("=" * 80)

    # Example 1: Related sentences (should predict IS NEXT)
    print("\n--- Example 1: Related Sentences ---")
    predict_next_sentence(
        model, tokenizer,
        "The weather is beautiful today.",
        "We should go for a walk in the park.",
        config
    )

    # Example 2: Unrelated sentences (should predict NOT NEXT)
    print("\n--- Example 2: Unrelated Sentences ---")
    predict_next_sentence(
        model, tokenizer,
        "The weather is beautiful today.",
        "Python is a programming language.",
        config
    )

    # Example 3: Sequential sentences
    print("\n--- Example 3: Sequential Narrative ---")
    predict_next_sentence(
        model, tokenizer,
        "Scientists have discovered a new species in the Amazon rainforest.",
        "The species appears to be a type of tree frog with unique coloring.",
        config
    )

    # Example 4: Non-sequential
    print("\n--- Example 4: Non-Sequential ---")
    predict_next_sentence(
        model, tokenizer,
        "Scientists have discovered a new species in the Amazon rainforest.",
        "Basketball is a popular sport in many countries.",
        config
    )

    # === Interactive mode ===
    print("\n\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("\nYou can now test the model with your own inputs!")
    print("(Type 'quit' to exit)")

    while True:
        print("\n" + "-" * 80)
        print("Choose an option:")
        print("1. Test Masked Language Modeling (MLM)")
        print("2. Test Next Sentence Prediction (NSP)")
        print("3. Quit")

        choice = input("\nEnter choice (1-3): ").strip()

        if choice == '1':
            text = input("\nEnter a sentence to mask: ").strip()
            if text:
                try:
                    num_masks = int(input("How many tokens to mask? (default: 2): ").strip() or "2")
                    predict_masked_tokens(model, tokenizer, text, config, num_masks)
                except Exception as e:
                    print(f"Error: {e}")

        elif choice == '2':
            sentence_a = input("\nEnter first sentence: ").strip()
            sentence_b = input("Enter second sentence: ").strip()
            if sentence_a and sentence_b:
                try:
                    print("\n" + "-" * 80)
                    predict_next_sentence(model, tokenizer, sentence_a, sentence_b, config)
                except Exception as e:
                    print(f"Error: {e}")

        elif choice == '3' or choice.lower() == 'quit':
            print("\nGoodbye!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
