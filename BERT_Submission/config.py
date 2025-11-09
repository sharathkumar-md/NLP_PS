import torch

class BERTConfig:
    """Configuration class for BERT model"""

    def __init__(self):
        # Model architecture
        self.vocab_size = 30522  # Same as BERT tokenizer
        self.hidden_size = 256  
        self.num_hidden_layers = 4  
        self.num_attention_heads = 4  
        self.intermediate_size = 1024  # Feed-forward hidden size (4 * hidden_size)
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 2  # For segment embeddings (sentence A vs B)
        self.layer_norm_eps = 1e-12

        # Training parameters 
        self.batch_size = 32  
        self.max_seq_length = 256  
        self.learning_rate = 2e-4  
        self.num_epochs = 10  
        self.warmup_steps = 1500
        self.max_steps = 20000 

        # MLM parameters
        self.mlm_probability = 0.15  # Probability of masking a token
        self.mask_token_prob = 0.8  # 80% replace with [MASK]
        self.random_token_prob = 0.1  # 10% replace with random token
        self.unchanged_prob = 0.1  # 10% keep unchanged

        # Special tokens
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.mask_token_id = 103

        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 

        # Logging
        self.log_interval = 100
        self.save_interval = 1000

    def __repr__(self):
        return f"BERTConfig(hidden_size={self.hidden_size}, num_layers={self.num_hidden_layers})"
