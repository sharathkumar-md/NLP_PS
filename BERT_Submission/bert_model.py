import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BERTEmbeddings(nn.Module):
    """
    BERT embeddings combine three types of embeddings:
    1. Token embeddings: vocabulary tokens
    2. Position embeddings: position in sequence
    3. Segment embeddings: which sentence (A or B) the token belongs to
    """

    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        """
        Args:
            input_ids: (batch_size, seq_length)
            token_type_ids: (batch_size, seq_length) - 0 for sentence A, 1 for sentence B
        Returns:
            embeddings: (batch_size, seq_length, hidden_size)
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        segment_embeds = self.segment_embeddings(token_type_ids)

        embeddings = token_embeds + position_embeds + segment_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism as described in "Attention Is All You Need"
    """

    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """
        Reshape from (batch_size, seq_length, hidden_size) to
        (batch_size, num_heads, seq_length, head_size)
        """
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
            attention_mask: (batch_size, 1, 1, seq_length) - 1 for tokens to attend, 0 for padding
        Returns:
            context: (batch_size, seq_length, hidden_size)
        """
        # Linear projections
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask (if provided)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context = torch.matmul(attention_probs, value_layer)

        # Reshape back
        context = context.permute(0, 2, 1, 3).contiguous()
        new_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_shape)

        return context


class BERTSelfOutput(nn.Module):
    """
    Apply linear transformation, dropout, and layer normalization after attention
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # Residual connection
        return hidden_states


class BERTAttention(nn.Module):
    """
    Complete attention block: self-attention + output transformation
    """

    def __init__(self, config):
        super().__init__()
        self.self = MultiHeadSelfAttention(config)
        self.output = BERTSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output


class BERTIntermediate(nn.Module):
    """
    Feed-forward intermediate layer with GELU activation
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)  # BERT uses GELU activation
        return hidden_states


class BERTOutput(nn.Module):
    """
    Feed-forward output layer with residual connection and layer normalization
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # Residual connection
        return hidden_states


class BERTLayer(nn.Module):
    """
    Single BERT transformer layer: attention + feed-forward network
    """

    def __init__(self, config):
        super().__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        # Self-attention
        attention_output = self.attention(hidden_states, attention_mask)

        # Feed-forward
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output


class BERTEncoder(nn.Module):
    """
    Stack of BERT transformer layers
    """

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([BERTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BERTPooler(nn.Module):
    """
    Pool the hidden states of the [CLS] token for sentence-level tasks
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # Take the hidden state of the first token ([CLS])
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BERTModel(nn.Module):
    """
    Main BERT model (encoder only)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_length)
            token_type_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length) - 1 for real tokens, 0 for padding
        Returns:
            sequence_output: (batch_size, seq_length, hidden_size) - all token representations
            pooled_output: (batch_size, hidden_size) - [CLS] token representation
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Create extended attention mask for attention layers
        # Convert from (batch_size, seq_length) to (batch_size, 1, 1, seq_length)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Convert to float and create mask (0 for attend, -10000 for ignore)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Get embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # Pass through encoder
        sequence_output = self.encoder(embedding_output, extended_attention_mask)

        # Pool [CLS] token
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output


class BERTForPreTraining(nn.Module):
    """
    BERT model with MLM and NSP heads for pre-training
    """

    def __init__(self, config):
        super().__init__()
        self.bert = BERTModel(config)

        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.vocab_size)
        )

        # NSP head
        self.nsp_head = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        """
        Args:
            input_ids: (batch_size, seq_length)
            token_type_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)
            masked_lm_labels: (batch_size, seq_length) - token IDs for masked positions, -100 for others
            next_sentence_label: (batch_size,) - 0 or 1
        Returns:
            If labels provided: total_loss, mlm_loss, nsp_loss
            Otherwise: mlm_logits, nsp_logits
        """
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)

        # MLM predictions
        mlm_logits = self.mlm_head(sequence_output)

        # NSP predictions
        nsp_logits = self.nsp_head(pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            # Compute MLM loss
            loss_fct = nn.CrossEntropyLoss()  # -100 index is ignored
            mlm_loss = loss_fct(mlm_logits.view(-1, self.bert.config.vocab_size), masked_lm_labels.view(-1))

            # Compute NSP loss
            nsp_loss = loss_fct(nsp_logits, next_sentence_label)

            # Total loss
            total_loss = mlm_loss + nsp_loss

            return total_loss, mlm_loss, nsp_loss
        else:
            return mlm_logits, nsp_logits
