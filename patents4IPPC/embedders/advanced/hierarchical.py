from collections import OrderedDict
from pathlib import Path

from transformers import AutoTokenizer
from tqdm import trange
import numpy as np
import torch

from patents4IPPC.custom_models.hierarchical_transformer import (
    HierarchicalTransformer, utils
)
from patents4IPPC.embedders.base_embedder import BaseEmbedder


class HierarchicalTransformerEmbedder(BaseEmbedder):
    def __init__(self, path_to_pretrained_model):
        """Text embedder based on a Hierarchical transformer model.

        Args:
            path_to_pretrained_model (str): Path to a pre-trained 
              Hierarchical transformer model.
        """

        self.model = HierarchicalTransformer.from_pretrained(
            path_to_pretrained_model
        )
        tokenizer_dir = Path(path_to_pretrained_model) / "segment_transformer"
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @property
    def embedding_size(self):
        return self.model.document_embedder.embedding_size

    @torch.no_grad()
    def embed_documents(
        self, documents, batch_size=64, do_lowercase=False, show_progress=False
    ):
        texts = (
            documents if not do_lowercase else list(map(str.lower, documents))
        )
        # NOTE: We assume each document to be pre-segmented and that 
        # the segments are separated by a special sequence of 
        # characters, namely "[SEGMENT_SEP]".
        segmented_texts = [text.split("[SEGMENT_SEP]") for text in texts]
        n_documents = len(texts)
        embeddings = []
        for batch_start_idx in trange(
            0, n_documents, batch_size, disable=(not show_progress)
        ):
            batch_end_idx = min(batch_start_idx + batch_size, n_documents)
            documents_batch = OrderedDict({
                f"Document_{i:04d}": segments
                for i, segments in enumerate(
                    segmented_texts[batch_start_idx:batch_end_idx]
                )
            })
            encoded_segments, _ = \
                utils.prepare_inputs_for_hierarchical_transformer(
                    documents_batch,
                    self.tokenizer,
                    self.model.segment_transformer.config.max_position_embeddings
                )            
            encoded_segments = utils.move_encoded_inputs_to_device(
                encoded_segments, self.device
            )

            document_ids_and_n_segments = [
                (doc_id, len(segments))
                for doc_id, segments in documents_batch.items()
            ]
            output = self.model(encoded_segments, document_ids_and_n_segments)

            embeddings_batch = (
                output
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            embeddings.append(embeddings_batch)
        
        return np.vstack(embeddings)
