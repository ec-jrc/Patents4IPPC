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

    def embed_documents(
        self, documents, batch_size=64, do_lowercase=False, show_progress=False
    ):
        texts = (
            documents if not do_lowercase else list(map(str.lower, documents))
        )
        # NOTE: We assume each document to be pre-segmented and that 
        # the segments are separated by tab ("\t") characters.
        segmented_texts = [text.split("\t") for text in texts]
        n_documents = len(texts)
        embeddings = []
        for batch_start_idx in trange(
            0, n_documents, batch_size, disable=(not show_progress)
        ):
            batch_end_idx = min(batch_start_idx + batch_size, n_documents)
            encoded_segments, document_ids = \
                utils.prepare_inputs_for_hierarchical_transformer(
                    segmented_texts[batch_start_idx:batch_end_idx],
                    self.tokenizer
                )
            encoded_segments = utils.move_encoded_inputs_to_device(
                encoded_segments, self.device
            )
            output = self.model(encoded_segments, document_ids)

            embeddings_batch = (
                output
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            embeddings.append(embeddings_batch)
        
        return np.vstack(embeddings)
