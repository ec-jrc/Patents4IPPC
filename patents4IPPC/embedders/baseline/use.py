import tensorflow_hub as hub
from tqdm import trange
import numpy as np

from patents4IPPC.embedders.base_embedder import BaseEmbedder

class UniversalSentenceEncoderEmbedder(BaseEmbedder):
    
    def __init__(self, path_to_model_checkpoint):
        """Text embedder based on the Universal Sentence Encoder (USE)
        model.

        Args:
            path_to_model_checkpoint (str): Path to a pretrained model 
              in Tensorflow 2.0 Saved Model format.
        """

        self.model = hub.load(path_to_model_checkpoint)
        self._embedding_size = self.model(['Dummy text']).shape[1]

    @property
    def embedding_size(self):
        return self._embedding_size

    def embed_documents(
        self, documents, batch_size=64, do_lowercase=False, show_progress=False
    ):
        texts = (
            documents if not do_lowercase else list(map(str.lower, documents))
        )
        n_documents = len(texts)
        embeddings = []
        for batch_start_idx in trange(
            0, n_documents, batch_size, disable=(not show_progress)
        ):
            # NOTE: Do I need to split the documents into separate
            #       sentences first? In that case, "pySBD" is an option
            batch_end_idx = min(batch_start_idx + batch_size, n_documents)
            embeddings_batch = self.model(texts[batch_start_idx:batch_end_idx])
            embeddings.append(embeddings_batch.numpy().astype(np.float32))
        return np.vstack(embeddings)
