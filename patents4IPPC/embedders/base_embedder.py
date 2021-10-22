from abc import ABC, abstractmethod

class BaseEmbedder(ABC):

    @property
    @abstractmethod
    def embedding_size(self):
        raise NotImplementedError(
            'Property `embedding_size` has no default implementation.'
        )
    
    @abstractmethod
    def embed_documents(
        self, documents, batch_size=64, do_lowercase=False, show_progress=False
    ):
        """Embed a list of text documents.

        Args:
            documents (list[str]): List of texts to embed.
            batch_size (int, optional): Batch size for embedding 
              `documents`. Defaults to 64.
            do_lowercase (bool, optional): Lowercase the texts before 
              embedding them. Defaults to False.
            show_progress (bool, optional): Show a progress bar while 
              embedding the texts. Defaults to False.

        Returns:
            numpy.array: NumPy array of shape (n_documents, embedding_size) 
              whose rows represent the embeddings of each document in 
              `documents.`

        """

        raise NotImplementedError(
            'Method `embed_documents` has no default implementation.'
        )
