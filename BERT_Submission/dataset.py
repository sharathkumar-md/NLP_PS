"""
Dataset preparation for BERT pre-training
Implements Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) data preparation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import random
import numpy as np


class BERTDataset(Dataset):
    """
    Dataset for BERT pre-training with MLM and NSP tasks
    """

    def __init__(self, config, tokenizer, split='train'):
        self.config = config
        self.tokenizer = tokenizer
        self.max_seq_length = config.max_seq_length

        # Load WikiText-2 dataset
        print(f"Loading WikiText-2 {split} dataset...")
        dataset = load_dataset('wikitext', 'wikitext-2-v1', split=split)

        # Process into sentence pairs
        print("Processing sentences...")
        self.examples = self._create_examples(dataset)
        print(f"Created {len(self.examples)} training examples")

    def _create_examples(self, dataset):
        """
        Create sentence pair examples from the dataset
        Each example is a tuple: (sentence_a, sentence_b, is_next)
        """
        examples = []

        # Extract all non-empty sentences
        sentences = []
        for item in dataset:
            text = item['text'].strip()
            if text and len(text) > 10:  # Filter out very short lines
                # Simple sentence splitting (not perfect but works for WikiText)
                sents = text.split('.')
                for sent in sents:
                    sent = sent.strip()
                    if len(sent) > 10:
                        sentences.append(sent)

        # Create sentence pairs
        i = 0
        while i < len(sentences) - 1:
            sentence_a = sentences[i]

            # 50% of the time, use the actual next sentence (is_next = 1)
            # 50% of the time, use a random sentence (is_next = 0)
            if random.random() < 0.5:
                sentence_b = sentences[i + 1]
                is_next = 1
                i += 2  # Move forward by 2 sentences
            else:
                # Pick a random sentence
                random_idx = random.randint(0, len(sentences) - 1)
                while random_idx == i or random_idx == i + 1:  # Avoid picking adjacent sentences
                    random_idx = random.randint(0, len(sentences) - 1)
                sentence_b = sentences[random_idx]
                is_next = 0
                i += 1  # Move forward by 1 sentence

            examples.append((sentence_a, sentence_b, is_next))

        return examples

    def _create_masked_lm_predictions(self, tokens):
        """
        Implement BERT's masking strategy:
        - 15% of tokens are selected for masking
        - Of the selected tokens:
          - 80% are replaced with [MASK]
          - 10% are replaced with a random token
          - 10% are kept unchanged

        Returns:
            output_tokens: tokens with masking applied
            masked_lm_labels: labels for masked positions (-100 for non-masked)
        """
        output_tokens = tokens.copy()
        masked_lm_labels = [-100] * len(tokens)  # -100 is ignored by CrossEntropyLoss

        # Don't mask special tokens [CLS], [SEP], [PAD]
        special_tokens = {self.config.cls_token_id, self.config.sep_token_id, self.config.pad_token_id}

        # Create list of maskable positions
        maskable_positions = [i for i, token_id in enumerate(tokens) if token_id not in special_tokens]

        # Randomly select 15% of positions for masking
        num_to_mask = max(1, int(len(maskable_positions) * self.config.mlm_probability))
        masked_positions = random.sample(maskable_positions, min(num_to_mask, len(maskable_positions)))

        for pos in masked_positions:
            masked_lm_labels[pos] = tokens[pos]  # Store original token as label

            # Apply masking strategy
            rand = random.random()
            if rand < self.config.mask_token_prob:  # 80%: replace with [MASK]
                output_tokens[pos] = self.config.mask_token_id
            elif rand < self.config.mask_token_prob + self.config.random_token_prob:  # 10%: random token
                output_tokens[pos] = random.randint(0, self.config.vocab_size - 1)
            # else 10%: keep original token

        return output_tokens, masked_lm_labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Get a single training example with MLM and NSP labels
        """
        sentence_a, sentence_b, is_next_label = self.examples[idx]

        # Tokenize sentences
        tokens_a = self.tokenizer.encode(sentence_a, add_special_tokens=False)
        tokens_b = self.tokenizer.encode(sentence_b, add_special_tokens=False)

        # Truncate if too long
        # Account for [CLS], [SEP], [SEP]
        max_tokens_per_sentence = (self.max_seq_length - 3) // 2
        if len(tokens_a) > max_tokens_per_sentence:
            tokens_a = tokens_a[:max_tokens_per_sentence]
        if len(tokens_b) > max_tokens_per_sentence:
            tokens_b = tokens_b[:max_tokens_per_sentence]

        # Combine into BERT input: [CLS] + tokens_a + [SEP] + tokens_b + [SEP]
        tokens = [self.config.cls_token_id] + tokens_a + [self.config.sep_token_id] + tokens_b + [self.config.sep_token_id]

        # Create segment (token type) IDs: 0 for sentence A, 1 for sentence B
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

        # Apply MLM masking
        masked_tokens, masked_lm_labels = self._create_masked_lm_predictions(tokens)

        # Pad to max_seq_length
        padding_length = self.max_seq_length - len(masked_tokens)
        input_ids = masked_tokens + [self.config.pad_token_id] * padding_length
        segment_ids = segment_ids + [0] * padding_length
        attention_mask = [1] * len(masked_tokens) + [0] * padding_length
        masked_lm_labels = masked_lm_labels + [-100] * padding_length

        # Convert to tensors
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(segment_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'masked_lm_labels': torch.tensor(masked_lm_labels, dtype=torch.long),
            'next_sentence_label': torch.tensor(is_next_label, dtype=torch.long)
        }


def create_dataloaders(config):
    """
    Create train and validation dataloaders
    """
    # Use BERT's pretrained tokenizer (only for tokenization, not the model)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create datasets
    train_dataset = BERTDataset(config, tokenizer, split='train')
    val_dataset = BERTDataset(config, tokenizer, split='validation')

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    # Test dataset creation
    from config import BERTConfig

    config = BERTConfig()
    train_loader, val_loader, tokenizer = create_dataloaders(config)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Show a sample batch
    batch = next(iter(train_loader))
    print("\nSample batch:")
    for key, value in batch.items():
        print(f"{key}: {value.shape}")

    # Decode a sample
    print("\nSample input (first example):")
    input_ids = batch['input_ids'][0]
    masked_tokens = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(f"Masked input: {masked_tokens}")
    print(f"NSP label: {batch['next_sentence_label'][0].item()}")
