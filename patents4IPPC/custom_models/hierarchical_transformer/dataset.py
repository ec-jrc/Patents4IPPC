from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from .utils import (
    index_encoded_inputs, prepare_inputs_for_hierarchical_transformer
)


class DocumentSimilarityDataset(Dataset):
    def __init__(
        self,
        left_documents: OrderedDict,
        right_documents: OrderedDict,
        labels: List[Tuple[str, str, float]],
        tokenizer,
        model_max_length
    ):
        self.tokenizer = tokenizer

        all_documents = OrderedDict()
        all_documents.update(left_documents)
        all_documents.update(right_documents)
        encoded_segments, document_ids = \
            prepare_inputs_for_hierarchical_transformer(
                all_documents, self.tokenizer, model_max_length
            )

        left_segments_mask = np.array(
            [doc_id in left_documents for doc_id in document_ids]
        )
        right_segments_mask = np.array(
            [doc_id in right_documents for doc_id in document_ids]
        )

        self.left_encoded_segments = index_encoded_inputs(
            encoded_segments, left_segments_mask
        )
        self.left_document_ids = document_ids[left_segments_mask]

        self.right_encoded_segments = index_encoded_inputs(
            encoded_segments, right_segments_mask
        )
        self.right_document_ids = document_ids[right_segments_mask]

        self.labels = labels

    @classmethod
    def from_directory(cls, path_to_dir, tokenizer, model_max_length):
        dataset_dir = Path(path_to_dir)
        qrels = pd.read_csv(
            str(dataset_dir / "qrels.txt"),
            header=None,
            names=["topic_patent", "rel_patent", "label"]
        )
        left_documents, right_documents = OrderedDict(), OrderedDict()
        labels = []
        for _, row in qrels.iterrows():
            left_doc_rel_filename = Path(row["topic_patent"])
            right_doc_rel_filename = Path(row["rel_patent"])
            left_doc_file = dataset_dir / left_doc_rel_filename
            right_doc_file = dataset_dir / right_doc_rel_filename
            label = row["label"]

            left_doc_id = left_doc_rel_filename.stem
            if left_doc_id not in left_documents:
                left_documents[left_doc_id] = \
                    left_doc_file.read_text().split("\n")
            
            right_doc_id = right_doc_rel_filename.stem
            if right_doc_id not in right_documents:
                right_documents[right_doc_id] = \
                    right_doc_file.read_text().split("\n")
            
            labels.append((left_doc_id, right_doc_id, label))

        dataset = cls(
            left_documents,
            right_documents,
            labels,
            tokenizer,
            model_max_length
        )
        return dataset

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        left_doc_id, right_doc_id, target_value = self.labels[index]
        
        left_segments_mask = self.left_document_ids == left_doc_id
        right_segments_mask = self.right_document_ids == right_doc_id

        n_segments_in_left_doc = left_segments_mask.sum()
        n_segments_in_right_doc = right_segments_mask.sum()

        return (
            index_encoded_inputs(
                self.left_encoded_segments, left_segments_mask
            ),
            (left_doc_id, n_segments_in_left_doc),
            index_encoded_inputs(
                self.right_encoded_segments, right_segments_mask
            ),
            (right_doc_id, n_segments_in_right_doc),
            target_value
        )
        # ^ NOTE: In general, each single element or batch in this
        # dataset will be a tuple whose elements do NOT have compatible 
        # shapes, because each document may a different number of 
        # segments. Note that this is not a problem because 
        # HierarchicalTransformer was explicitly designed to handle this 
        # case.
