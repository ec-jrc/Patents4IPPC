from time import time

import numpy as np
import faiss

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
