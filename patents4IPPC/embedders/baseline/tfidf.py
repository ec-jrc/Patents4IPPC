from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm
import numpy as np
import joblib

from patents4IPPC.embedders.base_embedder import BaseEmbedder


class TfidfEmbedder(BaseEmbedder):

    def __init__(self, return_dense_embeddings=True, **kwargs):
        """Text embedder based on the TF-IDF (term frequency - inverse 
        document frequency) algorithm.

        Args:
            return_dense_embeddings (bool, optional): Return regular 
              NumPy arrays for the embeddings rather than sparse arrays. 
              Defaults to True.
            **kwargs: Additional keyword arguments for Scikit-learn's 
              TFidfVectorizer.
        """

        self.tfidf_vectorizer = TfidfVectorizer(**kwargs)
        self.return_dense_embeddings = return_dense_embeddings

    @staticmethod
    def from_pretrained(path_to_pickled_model):
        return joblib.load(path_to_pickled_model)

    def fit(self, documents, show_progress=False):
        self.tfidf_vectorizer.fit(tqdm(documents, disable=(not show_progress)))

    @property
    def embedding_size(self):
        check_is_fitted(
            self.tfidf_vectorizer,
            msg=('`embedding_size` is not defined when `TfidfEmbedder` is '
                 'not fitted.')
        )
        return len(self.tfidf_vectorizer.vocabulary_)

    def embed_documents(
        self, documents, batch_size=64, do_lowercase=False, show_progress=False
    ):        
        sparse_embeddings = self.tfidf_vectorizer.transform(
            tqdm(documents, disable=(not show_progress))
        ).astype(np.float32)
        if not self.return_dense_embeddings:
            return sparse_embeddings
        dense_embeddings = sparse_embeddings.toarray()
        return dense_embeddings
