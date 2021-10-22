from pathlib import Path
import string
import math

import pandas as pd
import numpy as np
import joblib
import torch
import faiss

from transformers import BertTokenizerFast

from patents4IPPC.embedders.base_embedder import BaseEmbedder
from patents4IPPC.embedders.baseline import tfidf, glove, use
from patents4IPPC.embedders.advanced import huggingface
from patents4IPPC import preprocessing


def load_dataset(
    path_to_dataset,
    train_portion=None,
    valid_portion=None,
    test_portion=None,
    seed=None,
    normalize_labels=True
):
    dataset = pd.read_csv(path_to_dataset).dropna(subset=['query', 'response'])
    if normalize_labels:
        dataset.loc[:, 'label'] = dataset['label'] / dataset['label'].max()
    
    if 'split' in dataset.columns:
        dataset_train = dataset[dataset['split'] == 'train']
        dataset_valid = dataset[dataset['split'] == 'dev']
        dataset_test = dataset[dataset['split'] == 'test']
    else:
        assert train_portion + valid_portion + test_portion == 1.0, \
               'Fractions of train, validation and test do not sum up to 1.'
        # Get the unique query ids
        query_ids = dataset['query_id'].unique()
        # Shuffle them
        np.random.seed(seed)
        np.random.shuffle(query_ids)
        # Pick train, validation and test portions
        n_queries = len(query_ids)
        n_train_queries = math.ceil(n_queries * train_portion)

        nontrain_portion = 1 - train_portion
        rescaled_valid_portion = (0 if nontrain_portion == 0
                                  else valid_portion / nontrain_portion)
        rescaled_test_portion = (0 if nontrain_portion == 0
                                 else test_portion / nontrain_portion)
        n_nontrain_queries = math.floor(n_queries * nontrain_portion)
        n_valid_queries = math.ceil(n_nontrain_queries * rescaled_valid_portion)
        n_test_queries = math.floor(n_nontrain_queries * rescaled_test_portion)

        train_end_idx = n_train_queries
        train_ids = query_ids[:train_end_idx]

        valid_end_idx = train_end_idx + n_valid_queries
        valid_ids = query_ids[train_end_idx:valid_end_idx]

        test_end_idx = valid_end_idx + n_test_queries
        test_ids = query_ids[valid_end_idx:test_end_idx]

        dataset_train = dataset[dataset['query_id'].isin(train_ids)]
        dataset_valid = dataset[dataset['query_id'].isin(valid_ids)]
        dataset_test = dataset[dataset['query_id'].isin(test_ids)]

    return dataset_train, dataset_valid, dataset_test

def index_documents(
    documents,
    ids,
    embedder: BaseEmbedder,
    batch_size=64,
    do_lowercase=False,
    as_python_dictionary=False,
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
    if as_python_dictionary:
        index = dict(zip(ids, embeddings))
        # NOTE: The embeddings are not normalized in this case. That's
        # because the dictionary mode is used to train a DualTransformer
        # model, meaning that the normalization step is embedded in the
        # computation of the cosine similarity between query and
        # response embeddings within the loss function
        # TODO: You need to explicitly normalize embeddings if using
        # another loss function that doesn't involve computing the
        # cosine similarity
    else:
        # Create a FAISS index
        index = faiss.index_factory(
            embedder.embedding_size, 'IDMap,Flat', faiss.METRIC_INNER_PRODUCT
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
    if as_python_dictionary:
        joblib.dump(index, filename)
    else:
        faiss.write_index(index, filename)

def index_dummy_documents(
    n_documents, embedding_size, store_on_disk=False, filename=None
):
    # Create dummy embeddings as well as their IDs
    embeddings = np.random.rand(n_documents, embedding_size).astype(np.float32)
    ids = np.arange(n_documents)
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
    faiss.write_index(index, filename)

def get_embedder(model_type, path_to_model_checkpoint=None):
    if model_type == 'tfidf':
        if path_to_model_checkpoint is not None:
            embedder = tfidf.TfidfEmbedder.from_pretrained(
                path_to_model_checkpoint
            )
        else:
            embedder = tfidf.TfidfEmbedder(
                return_dense_embeddings=True,
                lowercase=True, # To increase the variety of possible words
                preprocessor=preprocessing.preprocess_for_tfidf,
                tokenizer=BertTokenizerFast.from_pretrained(
                    'bert-base-uncased'
                ).tokenize,
                max_df=0.7,
                max_features=8192 # TODO: Is this the right number?
            )
    elif model_type == 'glove':
        embedder = glove.GloveEmbedder(path_to_model_checkpoint)
    elif model_type == 'use':
        embedder = use.UniversalSentenceEncoderEmbedder(
            path_to_model_checkpoint
        )
    elif model_type == 'huggingface':
        embedder = huggingface.HuggingFaceTransformerEmbedder(
            path_to_model_checkpoint
        )
    elif model_type == 'dual':
        embedder = huggingface.DualTransformerEmbedder(
            path_to_model_checkpoint
        )
    else:
        raise ValueError(f'Unknown model type "{model_type}".')

    return embedder

def mean_pool(huggingface_model_output, attention_mask):
    output = huggingface_model_output.last_hidden_state
    pad_tokens_mask = torch.unsqueeze(attention_mask, -1)
    masked_output = output * pad_tokens_mask
    avg_pooled_output = (masked_output.sum(axis=1)
                         / pad_tokens_mask.sum(axis=1))
    return avg_pooled_output    

def ucid_to_int(ucid):
    capital_letters = string.ascii_uppercase
    ucid_no_dashes = ucid.replace('-', '')
    characters = list(ucid_no_dashes)
    encoded_characters = list(map(
        lambda c: str(capital_letters.index(c)) if c in capital_letters else c,
        characters
    ))
    return int(''.join(encoded_characters))
