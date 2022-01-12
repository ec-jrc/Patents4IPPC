import math
from functools import reduce
from operator import add
from typing import Dict, List

import numpy as np
import torch

from transformers import PreTrainedTokenizer

from patents4IPPC.custom_models.hierarchical_transformer.embedding_documents \
    import (
        DocumentEmbedderType,
        RecurrenceBasedDocumentEmbedderConfig,
        TransformerBasedDocumentEmbedderConfig
    )


def reshape_encoded_inputs(encoded_inputs, new_shape):
    return {k: v.reshape(new_shape) for k, v in encoded_inputs.items()}

def index_encoded_inputs(encoded_inputs, index):
    return {k: v[index] for k, v in encoded_inputs.items()}

def concat_encoded_inputs(encoded_inputs, dim=0):
    dict_keys = encoded_inputs[0].keys()
    return {
        k: torch.concat([enc_input[k] for enc_input in encoded_inputs], dim=dim)
        for k in dict_keys
    }

def move_encoded_inputs_to_device(encoded_inputs, device):
    return {k: v.to(device) for k, v in encoded_inputs.items()}

def prepare_inputs_for_hierarchical_transformer(
    documents: List[List[str]],
    tokenizer: PreTrainedTokenizer,
    model_max_length: int
):
    """Converts a list of documents into a format that's suitable for
    consumption by a HierarchicalTransformer model.

    Args:
        documents (List[List[str]]): List of documents, where each
          document is a list of segments (sentences, paragraphs or
          whatever).
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer for 
          tokenizing the segments.
        model_max_length (int): Maximum number of tokens that the 
          segment transformer can handle.
    
    Returns:
        Tuple[Dict, numpy.ndarray]: The tokenized segments and an array
          of integer IDs that associates each segment to the document it
          belongs to.
    """

    segments = list(reduce(add, documents))
    encoded_segments = tokenizer(
        segments,
        padding="max_length",
        max_length=model_max_length,
        truncation=True,
        return_tensors="pt"        
    ).data
    document_ids = []
    for i, document in enumerate(documents):
        document_ids.extend([i] * len(document))

    return encoded_segments, np.array(document_ids)

def build_sine_cosine_position_embeddings(max_len, embedding_size):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embedding_size, 2)
        * (-math.log(10000.0) / embedding_size)
    )
    position_embeddings = torch.zeros(1, max_len, embedding_size)
    position_embeddings[0, :, 0::2] = torch.sin(position * div_term)
    position_embeddings[0, :, 1::2] = torch.cos(position * div_term)
    return position_embeddings

def get_config_class_from_config_params(config_params: Dict):
    if "hidden_size" in config_params:
        return RecurrenceBasedDocumentEmbedderConfig
    if "attention_heads" in config_params:
        return TransformerBasedDocumentEmbedderConfig
    
    raise ValueError(f"Unknown configuration parameters: {config_params}.")

def get_embedder_type_from_config_params(config_params: Dict):
    if "hidden_size" in config_params:
        return DocumentEmbedderType.RECURRENCE
    if "attention_heads" in config_params:
        return DocumentEmbedderType.TRANSFORMER
    
    raise ValueError(f"Unknown configuration parameters: {config_params}.")
