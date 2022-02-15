from dataclasses import asdict
from pathlib import Path
import json
from typing import List, Tuple

from transformers import AutoModel
import numpy as np
import torch
import torch.nn.functional as F

from patents4IPPC.custom_models.hierarchical_transformer.embedding_documents import (
    DEFAULT_DOCUMENT_EMBEDDER_CONFIG,
    DocumentEmbedderType,
    RecurrenceBasedDocumentEmbedderConfig,
    TransformerBasedDocumentEmbedderConfig,
    get_document_embedder
)
from .utils import (
    index_encoded_inputs,
    get_config_class_from_config_params,
    get_embedder_type_from_config_params
)
from utils import pool_embeddings_with_attention_mask


class HierarchicalTransformer(torch.nn.Module):
    def __init__(
        self,
        segment_transformer,
        document_embedder_type=DocumentEmbedderType.TRANSFORMER,
        document_embedder_config=DEFAULT_DOCUMENT_EMBEDDER_CONFIG,
        segment_transformer_inner_batch_size=2,
        # ^ NOTE: This is NOT the effective batch size for the segment
        # transformer, but rather the amount of samples on which it can
        # perform a forward pass without incurring in an OOM error.
        segment_transformer_pooling_mode="mean"
    ):
        super().__init__()
        self.segment_transformer = segment_transformer
        self.document_embedder = get_document_embedder(
            document_embedder_type,
            segment_transformer.config,
            document_embedder_config
        )
        self.segment_transformer_inner_batch_size = \
            segment_transformer_inner_batch_size
        self.segment_transformer_pooling_mode = \
            segment_transformer_pooling_mode            

    def forward(
        self,
        encoded_inputs,
        document_ids_and_n_segments: List[Tuple[str, int]],
        return_segment_embeddings=False
    ):
        # `encoded_inputs` has shape (n_total_segments, segment_length),
        # where `n_total_segments` is the total number of segments in
        # each document.        
        segment_embeddings = self.get_segment_embeddings(encoded_inputs)
        # ^ (n_total_segments, embedding_size)
        document_embeddings = self.get_document_embeddings(
            segment_embeddings, document_ids_and_n_segments
        )
        # ^ (n_documents, embedding_size)

        if return_segment_embeddings:
            return segment_embeddings, document_embeddings
        return document_embeddings

    def get_segment_embeddings(self, encoded_inputs):
        token_embeddings = self._get_token_embeddings(encoded_inputs)
        # ^ (n_total_segments, segment_length, embedding_size)
        segment_embeddings = pool_embeddings_with_attention_mask(
            embeddings=token_embeddings,
            attention_mask=encoded_inputs["attention_mask"],
            mode=self.segment_transformer_pooling_mode
        )
        # ^ (n_total_segments, embedding_size)
        return segment_embeddings        

    def get_document_embeddings(self, segment_embeddings, document_ids_and_n_segments):
        batched_segment_embeddings, attention_mask = \
            self._separate_and_batch_segment_embeddings(
                segment_embeddings, document_ids_and_n_segments
            )
        # ^ `batched_segment_embeddings` has shape (n_documents, n_segments, embedding_size)
        # ^ `attention_mask` has shape (n_documents, n_segments)

        document_embeddings = self.document_embedder(
            batched_segment_embeddings, attention_mask
        )
        # ^ NOTE: Mean pooling happens within the document embedder's
        # `forward` method

        return document_embeddings

    def _get_token_embeddings(self, encoded_inputs):
        # NOTE: we do the forward propagation in the segment transformer 
        # one small batch at a time because, in principle, a document 
        # may contain a lot of segments. For example, 20 segments might 
        # already be too much in terms of memory requirements for a 
        # BERT-like model
        token_embeddings_batches = []
        for start_index in range(
            0,
            len(encoded_inputs["input_ids"]),
            self.segment_transformer_inner_batch_size
        ):
            end_index = start_index + self.segment_transformer_inner_batch_size
            encoded_inputs_batch = index_encoded_inputs(
                encoded_inputs, slice(start_index, end_index)
            )
            outputs_batch = self.segment_transformer(**encoded_inputs_batch)
            token_embeddings_batch = outputs_batch.last_hidden_state
            token_embeddings_batches.append(token_embeddings_batch)

        token_embeddings = torch.concat(token_embeddings_batches, dim=0)
        return token_embeddings

    def _separate_and_batch_segment_embeddings(
        self, segment_embeddings, document_ids_and_n_segments
    ):
        separated_segment_embeddings, attention_masks = \
            self._separate_segment_embeddings(
                segment_embeddings, document_ids_and_n_segments
            )

        batched_segment_embeddings, attention_mask = \
            self._batch_segment_embeddings(
                separated_segment_embeddings, attention_masks
            )
        
        return batched_segment_embeddings, attention_mask

    def _separate_segment_embeddings(
        self, segment_embeddings, document_ids_and_n_segments
    ):
        max_n_segments_in_document = max(
            n_segments for _, n_segments in document_ids_and_n_segments
        )
        padded_segment_embeddings = []
        attention_masks = []
        last_index = 0
        for _, n_segments in document_ids_and_n_segments:
            segment_embeddings_of_this_document = segment_embeddings[
                last_index:(last_index + n_segments)
            ]
            amount_of_pad_needed = max_n_segments_in_document - n_segments
            padded_segment_embeddings_of_this_document = F.pad(
                segment_embeddings_of_this_document,
                (0, 0, 0, amount_of_pad_needed),
                mode="constant",
                value=0.0
            )
            padded_segment_embeddings.append(
                padded_segment_embeddings_of_this_document
            )
            attention_masks.append(
                [1] * n_segments + [0] * amount_of_pad_needed
            )
            last_index += n_segments

        return padded_segment_embeddings, attention_masks

    def _batch_segment_embeddings(
        self, separated_segment_embeddings, attention_masks
    ):
        batched_segment_embeddings = torch.stack(
            separated_segment_embeddings, dim=0
        )
        attention_mask = torch.tensor(
            np.stack(attention_masks, axis=0),
            dtype=bool,
            device=batched_segment_embeddings.device.type
        )
        return batched_segment_embeddings, attention_mask

    @staticmethod
    def from_pretrained(
        path_to_checkpoint, segment_transformer_inner_batch_size=2
    ):
        checkpoint_dir = Path(path_to_checkpoint)

        segment_transformer_config_file = \
            (checkpoint_dir
             / "segment_transformer"
             / "segment_transformer_config.json")
        segment_transformer_config_params = \
            json.loads(segment_transformer_config_file.read_text())
        segment_transformer_kwargs = {
            f"segment_transformer_{p_name}": p_value
            for p_name, p_value in segment_transformer_config_params.items()
        }
        
        segment_transformer = AutoModel.from_pretrained(
            str(checkpoint_dir / "segment_transformer")
        )

        config_file = checkpoint_dir / "document_embedder" / "config.json"
        config_params = json.loads(config_file.read_text())

        document_embedder_type = get_embedder_type_from_config_params(
            config_params
        )
        ConfigClass = get_config_class_from_config_params(config_params)      
        document_embedder_config = ConfigClass(**config_params)
        model = HierarchicalTransformer(
            segment_transformer,
            document_embedder_type,
            document_embedder_config,
            segment_transformer_inner_batch_size,
            **segment_transformer_kwargs
        )

        state_dict = torch.load(
            str(checkpoint_dir / "document_embedder" / "document_embedder.pth")
        )
        model.document_embedder.load_state_dict(state_dict)
        return model.eval()  # Return in evaluation mode

    def save_pretrained(self, output_dir):
        checkpoint_dir = Path(output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._save_segment_transformer(checkpoint_dir)
        self._save_document_embedder(checkpoint_dir)

    def _save_segment_transformer(self, checkpoint_dir):
        segment_transformer_dir = checkpoint_dir / "segment_transformer"
        
        self.segment_transformer.save_pretrained(str(segment_transformer_dir))

        segment_transformer_config_file = \
            segment_transformer_dir / "segment_transformer_config.json"
        segment_transformer_config_file.write_text(
            json.dumps(
                {"pooling_mode": self.segment_transformer_pooling_mode},
                indent=2,
                sort_keys=True
            ) + "\n"
        )

    def _save_document_embedder(self, checkpoint_dir):
        document_embedder_dir = checkpoint_dir / "document_embedder"
        document_embedder_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.document_embedder.state_dict(),
            str(document_embedder_dir / "document_embedder.pth")
        )

        document_embedder_config_file = document_embedder_dir / "config.json"
        document_embedder_config_file.write_text(
            json.dumps(
                asdict(self.document_embedder.document_embedder_config),
                indent=2,
                sort_keys=True
            ) + "\n"
        )
