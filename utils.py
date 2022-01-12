import math
import string
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import torch

from patents4IPPC.embedders.base_embedder import BaseEmbedder


def load_dataset(
    path_to_dataset,
    train_portion=None,
    valid_portion=None,
    test_portion=None,
    seed=None,
    normalize_labels=True
):
    dataset = pd.read_csv(path_to_dataset).dropna(subset=["query", "response"])
    if normalize_labels:
        dataset.loc[:, "label"] = dataset["label"] / dataset["label"].max()
    
    if "split" in dataset.columns:
        dataset_train = dataset[dataset["split"] == "train"]
        dataset_valid = dataset[dataset["split"] == "dev"]
        dataset_test = dataset[dataset["split"] == "test"]
    else:
        assert train_portion + valid_portion + test_portion == 1.0, \
               "Fractions of train, validation and test do not sum up to 1."
        # Get the unique query ids
        query_ids = dataset["query_id"].unique()
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

        dataset_train = dataset[dataset["query_id"].isin(train_ids)]
        dataset_valid = dataset[dataset["query_id"].isin(valid_ids)]
        dataset_test = dataset[dataset["query_id"].isin(test_ids)]

    return dataset_train, dataset_valid, dataset_test

def index_documents_as_python_dictionary(
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

    index = dict(zip(ids, embeddings))
    # NOTE: The embeddings are not normalized in this case. That's
    # because the dictionary mode is used to train a DualTransformer
    # model, meaning that the normalization step is embedded in the
    # computation of the cosine similarity between query and
    # response embeddings within the loss function
    # TODO: You need to explicitly normalize embeddings if using
    # another loss function that doesn't involve computing the
    # cosine similarity

    if not store_on_disk:
        return index
    
    # Write the index to disk (can be loaded again later)
    if filename is None:
        raise ValueError(
            "A filename must be provided when you want to store an index on "
            "disk."
        )
    
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, filename)

# Taken from Huggingface Hub
def mean_pool_embeddings_with_attention_mask(embeddings, attention_mask):
    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)
        .expand(embeddings.size())
        .float()
    )
    return (torch.sum(embeddings * input_mask_expanded, axis=1)
            / torch.clamp(input_mask_expanded.sum(axis=1), min=1e-9))

def ucid_to_int(ucid):
    capital_letters = string.ascii_uppercase
    ucid_no_dashes = ucid.replace("-", "")
    characters = list(ucid_no_dashes)
    encoded_characters = list(map(
        lambda c: str(capital_letters.index(c)) if c in capital_letters else c,
        characters
    ))
    return int("".join(encoded_characters))
