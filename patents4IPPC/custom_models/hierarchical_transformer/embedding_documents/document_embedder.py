from abc import abstractmethod

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch

from patents4IPPC.custom_models.hierarchical_transformer.utils import (
    build_sine_cosine_position_embeddings
)
from utils import mean_pool_embeddings_with_attention_mask
from . import DocumentEmbedderType


class DocumentEmbedder(torch.nn.Module):
    def __init__(self, segment_transformer_config, document_embedder_config):
        super().__init__()
        self.segment_transformer_config = segment_transformer_config
        self.document_embedder_config = document_embedder_config
    
    def forward(self, segment_embeddings, attention_mask):
        transformed_segment_embeddings = self.transform_segment_embeddings(
            segment_embeddings, attention_mask
        )
        document_embeddings = mean_pool_embeddings_with_attention_mask(
            transformed_segment_embeddings, attention_mask
        )
        return document_embeddings

    # A template method in the terminology of the Template Method pattern
    @abstractmethod
    def transform_segment_embeddings(self, segment_embeddings, attention_mask):
        raise NotImplementedError(
            "Method `transform_segment_embeddings` is not implemented in "
            "class `DocumentEmbedder`."
        )
    
    @property
    @abstractmethod
    def embedding_size(self):
        raise NotImplementedError(
            "Property `embedding_size` is not implemented in class "
            "`DocumentEmbedder`."
        )


class RecurrenceBasedDocumentEmbedder(DocumentEmbedder):
    def __init__(self, segment_transformer_config, document_embedder_config):
        super().__init__(segment_transformer_config, document_embedder_config)
        self.lstm = torch.nn.LSTM(
            input_size=self.segment_transformer_config.hidden_size,
            hidden_size=self.document_embedder_config.hidden_size,
            num_layers=self.document_embedder_config.num_layers,
            batch_first=True,
            dropout=self.document_embedder_config.dropout_drop_rate
        )

    def transform_segment_embeddings(self, segment_embeddings, attention_mask):
        packed_segment_embeddings = pack_padded_sequence(
            segment_embeddings,
            lengths=attention_mask.sum(dim=1).cpu(),
            # ^ Tensor is required to be in the CPU
            batch_first=True,
            enforce_sorted=False
        )
        packed_and_transformed_segment_embeddings, final_hidden_states = \
            self.lstm(packed_segment_embeddings)
        # ^ Set the initial hidden states to 0 by not passing anything
        # as the second argument to `forward`.
        transformed_segment_embeddings, _ = pad_packed_sequence(
            packed_and_transformed_segment_embeddings,
            batch_first=True,
            padding_value=0.0
        )
        return transformed_segment_embeddings

    @property
    def embedding_size(self):
        return self.document_embedder_config.hidden_size


# Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(torch.nn.Module):
    def __init__(self, embedding_size, dropout_drop_rate=0.1, max_len=4096):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_drop_rate)
        position_embeddings = build_sine_cosine_position_embeddings(
            max_len, embedding_size
        )
        self.register_buffer("position_embeddings", position_embeddings)

    def forward(self, embeddings):
        # `embeddings` has shape (batch_size, seq_len, embedding_size)
        embeddings += self.position_embeddings[:, :embeddings.size(1)]
        return self.dropout(embeddings)
        # ^ We apply dropout as in the original Transformer paper


class TransformerBasedDocumentEmbedder(DocumentEmbedder):
    def __init__(self, segment_transformer_config, document_embedder_config):
        super().__init__(segment_transformer_config, document_embedder_config)
        self.segments_positional_encoder = PositionalEncoding(
            embedding_size=self.segment_transformer_config.hidden_size,
            dropout_drop_rate=self.document_embedder_config.dropout_drop_rate,
            max_len=self.segment_transformer_config.max_position_embeddings
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.segment_transformer_config.hidden_size,
            nhead=self.document_embedder_config.attention_heads,
            dim_feedforward=self.document_embedder_config.dim_feedforward,
            dropout=self.document_embedder_config.dropout_drop_rate,
            activation=self.document_embedder_config.activation,
            layer_norm_eps=self.document_embedder_config.layer_norm_eps,
            batch_first=True
        )
        self.document_transformer = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=self.document_embedder_config.num_layers
        )

    def transform_segment_embeddings(self, segment_embeddings, attention_mask):
        segment_embeddings_with_positional_encoding = \
            self.segments_positional_encoder(segment_embeddings)
        transformed_segment_embeddings = self.document_transformer(
            segment_embeddings_with_positional_encoding,
            src_key_padding_mask=~attention_mask
        )
        return transformed_segment_embeddings

    @property
    def embedding_size(self):
        return self.segment_transformer_config.hidden_size


def get_document_embedder(
    document_embedder_type,
    segment_transformer_config,
    document_embedder_config
):
    if document_embedder_type == DocumentEmbedderType.RECURRENCE:
        document_embedder = RecurrenceBasedDocumentEmbedder(
            segment_transformer_config, document_embedder_config
        )
    elif document_embedder_type == DocumentEmbedderType.TRANSFORMER:
        document_embedder = TransformerBasedDocumentEmbedder(
            segment_transformer_config, document_embedder_config
        )
    
    return document_embedder
