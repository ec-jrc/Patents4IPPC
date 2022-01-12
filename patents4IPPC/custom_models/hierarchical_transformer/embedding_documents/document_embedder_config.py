from dataclasses import dataclass
import json


class DocumentEmbedderConfig:
    def __init__(self, *args, **kwargs):
        raise TypeError(
            f"'{self.__class__.__name__}' cannot be directly instantiated."
        )
    
    @classmethod
    def from_json(cls, path_to_json_file):
        with open(path_to_json_file, "r") as f:
            config_params = json.load(f)
        return cls(**config_params)    

@dataclass
class RecurrenceBasedDocumentEmbedderConfig(DocumentEmbedderConfig):
    dropout_drop_rate: float
    hidden_size: int
    num_layers: int

@dataclass
class TransformerBasedDocumentEmbedderConfig(DocumentEmbedderConfig):
    activation: str
    attention_heads: int
    dim_feedforward: int
    dropout_drop_rate: float
    num_layers: int
    layer_norm_eps: float = 1e-05

DEFAULT_DOCUMENT_EMBEDDER_CONFIG = TransformerBasedDocumentEmbedderConfig(
    activation="gelu",
    attention_heads=8,
    dim_feedforward=2048,
    dropout_drop_rate=0.1,
    num_layers=2,
    layer_norm_eps=0.00001
)
