from pathlib import Path
import joblib

from stanfordcorenlp import StanfordCoreNLP
from gensim.models import KeyedVectors
from tqdm import tqdm
import numpy as np

from patents4IPPC.embedders.base_embedder import BaseEmbedder


class GloveEmbedder(BaseEmbedder):
    
    def __init__(self, path_to_pretrained_model):
        """Text embedder based on the GloVe (Global Vectors) algorithm.

        Args:
            path_to_pretrained_model (str): Path to pretrained GloVe 
              embeddings in a format that is suitable for use with 
              Gensim (e.g. ".gz").
        """

        base_dir = Path(path_to_pretrained_model)
        path_to_embeddings = next(base_dir.glob('*.gz'))
        path_to_stanford_corenlp = next(base_dir.glob('stanford-corenlp*'))
        self.model = KeyedVectors.load_word2vec_format(str(path_to_embeddings))
        self.tokenizer = \
            StanfordCoreNLP(str(path_to_stanford_corenlp)).word_tokenize
        self._embedding_size = self.model['the'].shape[0]

    @property
    def embedding_size(self):
        return self._embedding_size

    def _embed_single_document(
        self, document, do_lowercase=False
    ):
        text = document.lower() if do_lowercase else document
        tokens = self.tokenizer(text)
        embeddings = np.array([
            self.model[token] for token in tokens
            if token in self.model
        ])
        if len(embeddings) == 0:
            return np.zeros(self.embedding_size)
        average_embedding = np.mean(embeddings, axis=0).astype(np.float32)
        return average_embedding

    def embed_documents(
        self, documents, batch_size=64, do_lowercase=False, show_progress=False
    ):
        return np.stack([
            self._embed_single_document(doc, do_lowercase=do_lowercase)
            for doc in tqdm(documents, disable=(not show_progress))
        ])
