from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import List, Union

from captum.attr import LayerIntegratedGradients
from transformers import AutoTokenizer
import torch

from patents4IPPC.custom_models.hierarchical_transformer import (
    HierarchicalTransformer
)
from patents4IPPC.custom_models.hierarchical_transformer.utils import (
    move_encoded_inputs_to_device, prepare_inputs_for_hierarchical_transformer
)
from utils import pool_embeddings_with_attention_mask


class HierarchicalTransformerTextSimilarityExplainer:
    
    def __init__(self, path_to_model):
        self.model = HierarchicalTransformer.from_pretrained(
            path_to_model, segment_transformer_inner_batch_size=2
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            Path(path_to_model) / "segment_transformer"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model.to(self.device)

    def explain(
        self,
        text1: Union[str, List[str]],
        text2: Union[str, List[str]],
        n_steps: int = 50,
        internal_batch_size: int = 2,
        normalize_attributions: bool = False
    ):
        (
            text1_encoded_segments,
            text2_encoded_segments,
            baseline1_encoded_segments,
            baseline2_encoded_segments
        ) = self._construct_baselines_and_inputs(text1, text2)
        
        # Explain text1 while keeping text2 freezed
        with torch.no_grad():
            text2_embedding = self._embed_single_text(
                text2_encoded_segments["input_ids"],
                text2_encoded_segments["attention_mask"]
            )
        text1_token_attributions, text1_segment_attributions = \
            self._compute_attributions(
                text1_encoded_segments["input_ids"],
                baseline1_encoded_segments["input_ids"],
                text1_encoded_segments["attention_mask"],
                text2_embedding,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size
            )

        # Explain text2 while keeping text1 freezed
        with torch.no_grad():
            text1_embedding = self._embed_single_text(
                text1_encoded_segments["input_ids"],
                text1_encoded_segments["attention_mask"]
            )
        text2_token_attributions, text2_segment_attributions = \
            self._compute_attributions(
                text2_encoded_segments["input_ids"],
                baseline2_encoded_segments["input_ids"],
                text2_encoded_segments["attention_mask"],
                text1_embedding,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size
            )

        # Summarize and optionally normalize attributions
        text1_token_attributions = self._summarize_attributions(
            text1_token_attributions
        )
        text2_token_attributions = self._summarize_attributions(
            text2_token_attributions
        )
        text1_segment_attributions = self._summarize_attributions(
            text1_segment_attributions
        )
        text2_segment_attributions = self._summarize_attributions(
            text2_segment_attributions
        )        
        if normalize_attributions:
            text1_token_attributions = self._normalize_attributions(
                text1_token_attributions
            )
            text2_token_attributions = self._normalize_attributions(
                text2_token_attributions
            )
            text1_segment_attributions = self._normalize_attributions(
                text1_segment_attributions
            )
            text2_segment_attributions = self._normalize_attributions(
                text2_segment_attributions
            )

        text1_tokens = self.tokenizer.convert_ids_to_tokens(
            text1_encoded_segments["input_ids"].detach().cpu().tolist()
        )
        text2_tokens = self.tokenizer.convert_ids_to_tokens(
            text2_encoded_segments["input_ids"].detach().cpu().tolist()
        )

        text1_segments = \
            text1 if isinstance(text1, list) else text1.split("[SEGMENT_SEP]")
        text2_segments = \
            text2 if isinstance(text2, list) else text2.split("[SEGMENT_SEP]")            
        return (
            text1_tokens,
            text1_token_attributions.detach().cpu().tolist(),
            text1_segments,
            text1_segment_attributions.detach().cpu().tolist(),
            text2_tokens,
            text2_token_attributions.detach().cpu().tolist(),
            text2_segments,
            text2_segment_attributions.detach().cpu().tolist()
        )

    def _construct_baselines_and_inputs(self, text1, text2):
        segment_transformer_max_length = \
            self.model.segment_transformer.config.max_position_embeddings
        
        text1_encoded_segments, _ = prepare_inputs_for_hierarchical_transformer(
            OrderedDict({
                "text1": (text1 if isinstance(text1, list)
                          else text1.split("[SEGMENT_SEP]"))
            }),
            self.tokenizer,
            segment_transformer_max_length
        )
        text2_encoded_segments, _ = prepare_inputs_for_hierarchical_transformer(
            OrderedDict({
                "text2": (text2 if isinstance(text2, list)
                          else text2.split("[SEGMENT_SEP]"))
            }),
            self.tokenizer,
            segment_transformer_max_length
        )

        baseline1_encoded_segments = deepcopy(text1_encoded_segments)
        baseline1_encoded_segments["input_ids"].apply_(
            lambda token_id:
                token_id
                if token_id in self.tokenizer.all_special_ids
                else self.tokenizer.pad_token_id
        )
        
        baseline2_encoded_segments = deepcopy(text2_encoded_segments)
        baseline2_encoded_segments["input_ids"].apply_(
            lambda token_id:
                token_id
                if token_id in self.tokenizer.all_special_ids
                else self.tokenizer.pad_token_id
        )

        return (
            move_encoded_inputs_to_device(text1_encoded_segments, self.device),
            move_encoded_inputs_to_device(text2_encoded_segments, self.device),
            move_encoded_inputs_to_device(baseline1_encoded_segments, self.device),
            move_encoded_inputs_to_device(baseline2_encoded_segments, self.device)
        )

    def _embed_single_text(self, input_ids, attention_mask):
        document_ids_and_n_segments = [("text", len(input_ids))]
        text_embedding = self.model(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            document_ids_and_n_segments
        )
        return text_embedding

    def _compute_attributions(
        self,
        input_ids,
        baseline_ids,
        attention_mask,
        other_text_embedding,
        n_steps=50,
        internal_batch_size=2
    ):
        token_attributions = self._compute_token_attributions(
            input_ids,
            baseline_ids,
            attention_mask,
            other_text_embedding,
            n_steps,
            internal_batch_size
        )
        segment_attributions = self._compute_segment_attributions(
            input_ids,
            baseline_ids,
            attention_mask,
            other_text_embedding,
            n_steps,
            internal_batch_size
        )
        
        return token_attributions, segment_attributions

    def _summarize_attributions(self, attributions):
        attributions = attributions.sum(dim=-1)
        attributions = attributions / torch.norm(attributions)
        return attributions

    def _normalize_attributions(self, attributions):
        signs = torch.sign(attributions)
        attribution_magnitudes = torch.abs(attributions)
        normalized_attribution_magnitudes = (attribution_magnitudes
                                             / attribution_magnitudes.max())
        return signs * normalized_attribution_magnitudes   

    def _compute_token_attributions(
        self,
        input_ids,
        baseline_ids,
        attention_mask,
        other_text_embedding,
        n_steps,
        internal_batch_size
    ):
        self.model.zero_grad()

        attribution_engine = LayerIntegratedGradients(
            self._measure_cosine_similarity,
            self.model.segment_transformer.embeddings.word_embeddings
        )

        token_attributions = self._compute_integrated_gradients(
            attribution_engine,
            input_ids,
            baseline_ids,
            attention_mask,
            other_text_embedding,
            n_steps,
            internal_batch_size
        )

        return token_attributions     

    def _compute_segment_attributions(
        self,
        input_ids,
        baseline_ids,
        attention_mask,
        other_text_embedding,
        n_steps,
        internal_batch_size
    ):
        self.model.zero_grad()

        attribution_engine = LayerIntegratedGradients(
            self._measure_cosine_similarity, self.model.document_embedder
        )

        segment_attributions = self._compute_integrated_gradients(
            attribution_engine,
            input_ids,
            baseline_ids,
            attention_mask,
            other_text_embedding,
            n_steps,
            internal_batch_size
        )

        return segment_attributions

    def _measure_cosine_similarity(
        self, input_ids, attention_mask, other_text_embedding
    ):
        input_embedding = self._embed_single_text(input_ids, attention_mask)
        return torch.cosine_similarity(
            input_embedding, other_text_embedding, dim=0
        )

    def _compute_integrated_gradients(
        self,
        attribution_engine,
        input_ids,
        baseline_ids,
        attention_mask,
        other_text_embedding,
        n_steps,
        internal_batch_size        
    ):
        attributions, convergence_delta = attribution_engine.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask, other_text_embedding),
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=True
        )

        relative_convergence_delta = torch.abs(
            convergence_delta[0] / torch.sum(attributions)
        )
        if relative_convergence_delta > 0.05:
            print("WARNING: Relative convergence delta is > 5% "
                  f"({relative_convergence_delta*100:.2f}%). You may want to "
                  "recompute attributions passing a larger `n_steps` (current "
                  f"value is {n_steps}).")

        return attributions
