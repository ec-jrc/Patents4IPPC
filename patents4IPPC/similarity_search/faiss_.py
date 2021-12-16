from pathlib import Path
from time import time

import numpy as np
import faiss

from patents4IPPC.embedders.base_embedder import BaseEmbedder


def _create_index(
    embeddings, ids, embedding_size, store_on_disk=False, filename=None
):
    # Create a FAISS index
    index = faiss.index_factory(
        embedding_size, 'IDMap,Flat', faiss.METRIC_INNER_PRODUCT
    )
    # Add the normalized embeddings to the index. Normalization is
    # needed to compute cosine similarities when a search is performed.
    # Note that queries will also need to be normalized
    faiss.normalize_L2(embeddings)
    index.add_with_ids(embeddings, ids)

    if not store_on_disk:
        return index
    
    # Write the index to disk (can be loaded again later)
    if filename is None:
        raise ValueError(
            'A filename must be provided when you want to store an index on '
            'disk.'
        )
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, filename)    

def index_documents_using_faiss(
    documents,
    ids,
    embedder: BaseEmbedder,
    batch_size=64,
    do_lowercase=False,
    store_on_disk=False,
    filename=None
):
    # Embed the documents
    embeddings = embedder.embed_documents(
        documents,
        batch_size=batch_size,
        do_lowercase=do_lowercase,
        show_progress=True
    )
    # Create and possibly store the index on disk
    _create_index(
        embeddings, ids, embedder.embedding_size, store_on_disk, filename
    )

def index_dummy_documents_using_faiss(
    n_documents, embedding_size, store_on_disk=False, filename=None
):
    # Create dummy embeddings as well as their IDs
    embeddings = np.random.rand(n_documents, embedding_size).astype(np.float32)
    ids = np.arange(n_documents)
    # Create and possibly store the index on disk
    _create_index(
        embeddings, ids, embedding_size, store_on_disk, filename
    )    

def _log_message(message, verbose):
    if not verbose:
        return
    print(message)


class FaissDocumentRetriever(object):

    def __init__(self, index, use_gpu=False, gpu_id=0, verbose=False):
        self.verbose = verbose
        if isinstance(index, str):
            _log_message('Loading index from disk...', self.verbose)
            tstart = time()
            self.index = faiss.read_index(index) # faiss.IO_FLAG_MMAP
            tend = time()
            _log_message(
                f'Index loaded (took {tend-tstart:.4f} seconds).',
                self.verbose
            )
        else:
            self.index = index
        
        if use_gpu:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                device=gpu_id,
                index=self.index
            )

    def find_closest_matches(self, queries, k=1):
        # Normalize the queries. It is advised that we make a copy of
        # them as `faiss.normalize_L2` modifies them in place
        queries_copy = queries.copy().astype(np.float32)
        faiss.normalize_L2(queries_copy)
        # Find the most relevant documents with respect to each of the
        # given queries
        _log_message('Executing queries...', self.verbose)
        tstart = time()
        similarities, indices = self.index.search(queries_copy, k)
        tend = time()
        _log_message(
            f'Done executing queries (took {tend-tstart:.4f} seconds).',
            self.verbose
        )
        return similarities, indices
