from pathlib import Path
from typing import List

from torch.utils.data import Dataset
import pandas as pd

from .utils import (
    index_encoded_inputs, prepare_inputs_for_hierarchical_transformer
)


class DocumentSimilarityDataset(Dataset):
    def __init__(
        self,
        left_documents: List[List[str]],
        right_documents: List[List[str]],
        labels,
        tokenizer,
        model_max_length
    ):
        assert len(left_documents) == len(right_documents) == len(labels), \
            ("`left_documents`, `right_documents` and `labels` must all be "
             "the same length.")

        self.tokenizer = tokenizer

        all_documents = left_documents + right_documents
        encoded_segments, document_ids = \
            prepare_inputs_for_hierarchical_transformer(
                all_documents, self.tokenizer, model_max_length
            )

        left_segments_mask = (document_ids <= max(document_ids) // 2)
        right_segments_mask = (document_ids > max(document_ids) // 2)

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
        left_documents, right_documents, labels = [], [], []
        for _, row in qrels.iterrows():
            left_doc_file = dataset_dir / Path(row["topic_patent"])
            right_doc_file = dataset_dir / Path(row["rel_patent"])
            label = row["label"]

            left_documents.append(left_doc_file.read_text().split("\n"))
            right_documents.append(right_doc_file.read_text().split("\n"))
            labels.append(label)

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
        left_segments_mask = self.left_document_ids == index
        right_segments_mask = self.right_document_ids == (index + len(self.labels))
        # ^ We add `len(self.labels)` to `index` because, for example, 
        # index I in the left documents corresponds to index 
        # I + n_documents in the right documents

        return (
            index_encoded_inputs(
                self.left_encoded_segments, left_segments_mask
            ),
            self.left_document_ids[left_segments_mask],
            index_encoded_inputs(
                self.right_encoded_segments, right_segments_mask
            ),
            self.right_document_ids[right_segments_mask],
            self.labels[index]
        )
        # ^ NOTE: In general, each single element or batch in this
        # dataset will be a tuple whose elements do NOT have compatible 
        # shapes, except elements 1-2 and 3-4. This is because each
        # document has a different number of segments in general. Note
        # that this is not a problem because HierarchicalTransformer was
        # explicitly designed to handle this case
