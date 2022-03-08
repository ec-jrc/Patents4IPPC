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
from patents4IPPC.custom_models.hierarchical_transformer.embedding_documents import (
    RecurrenceBasedDocumentEmbedder
)
from patents4IPPC.custom_models.hierarchical_transformer.utils import (
    move_encoded_inputs_to_device, prepare_inputs_for_hierarchical_transformer
)


class HierarchicalTransformerTextSimilarityExplainer:
    
    def __init__(
        self,
        path_to_model,
        disable_gradients_computation_for_segment_transformer=False
    ):
        self.model = HierarchicalTransformer.from_pretrained(
            path_to_model, segment_transformer_inner_batch_size=1
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            Path(path_to_model) / "segment_transformer"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model.to(self.device)
        
        if disable_gradients_computation_for_segment_transformer:
            self.model.segment_transformer.requires_grad_(False)

    def explain(
        self,
        text1: Union[str, List[str]],
        text2: Union[str, List[str]],
        n_steps: int = 50,
        internal_batch_size: int = 2,
        normalize_attributions: bool = False
    ):
        (
            text1_segment_embeddings,
            text2_segment_embeddings,
            baseline1_segment_embeddings,
            baseline2_segment_embeddings
        ) = self._construct_baselines_and_inputs(text1, text2)
        
        # Explain text1 while keeping text2 frozen
        with torch.no_grad():
            text2_embedding = self._get_document_embedding(
                text2_segment_embeddings
            )
        text1_segment_attributions = \
            self._compute_attributions(
                text1_segment_embeddings,
                baseline1_segment_embeddings,
                text2_embedding,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size
            )

        # Explain text2 while keeping text1 frozen
        with torch.no_grad():
            text1_embedding = self._get_document_embedding(
                text1_segment_embeddings
            )
        text2_segment_attributions = \
            self._compute_attributions(
                text2_segment_embeddings,
                baseline2_segment_embeddings,
                text1_embedding,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size
            )

        # Summarize and optionally normalize attributions
        text1_segment_attributions = self._summarize_attributions(
            text1_segment_attributions
        )
        text2_segment_attributions = self._summarize_attributions(
            text2_segment_attributions
        )
        if normalize_attributions:
            text1_segment_attributions = self._normalize_attributions(
                text1_segment_attributions
            )
            text2_segment_attributions = self._normalize_attributions(
                text2_segment_attributions
            )

        text1_segments = \
            text1 if isinstance(text1, list) else text1.split("[SEGMENT_SEP]")
        text2_segments = \
            text2 if isinstance(text2, list) else text2.split("[SEGMENT_SEP]")            
        return (
            text1_segments,
            text1_segment_attributions.detach().cpu().tolist(),
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
        with torch.no_grad():
            text1_segment_embeddings = self.model.get_segment_embeddings(
                move_encoded_inputs_to_device(
                    text1_encoded_segments, self.device
                )
            )

        text2_encoded_segments, _ = prepare_inputs_for_hierarchical_transformer(
            OrderedDict({
                "text2": (text2 if isinstance(text2, list)
                          else text2.split("[SEGMENT_SEP]"))
            }),
            self.tokenizer,
            segment_transformer_max_length
        )
        with torch.no_grad():
            text2_segment_embeddings = self.model.get_segment_embeddings(
                move_encoded_inputs_to_device(
                    text2_encoded_segments, self.device
                )
            )

        baseline1_segment_embeddings = torch.zeros_like(
            text1_segment_embeddings
        )
        baseline2_segment_embeddings = torch.zeros_like(
            text2_segment_embeddings
        )

        ################################################################
        # TODO: Remove commented out code

        # baseline1_encoded_segments = deepcopy(text1_encoded_segments)
        # baseline1_encoded_segments["input_ids"].apply_(
        #     lambda token_id:
        #         token_id
        #         if token_id in self.tokenizer.all_special_ids
        #         else self.tokenizer.pad_token_id
        # )
        
        # baseline2_encoded_segments = deepcopy(text2_encoded_segments)
        # baseline2_encoded_segments["input_ids"].apply_(
        #     lambda token_id:
        #         token_id
        #         if token_id in self.tokenizer.all_special_ids
        #         else self.tokenizer.pad_token_id
        # )

        # def unsqueeze_encoded_inputs(encoded_inputs, dim=0):
        #     return {k: v.unsqueeze(dim) for k, v in encoded_inputs.items()}
        
        # return (
        #     move_encoded_inputs_to_device(
        #         unsqueeze_encoded_inputs(text1_encoded_segments, dim=0),
        #         self.device
        #     ),
        #     move_encoded_inputs_to_device(
        #         unsqueeze_encoded_inputs(text2_encoded_segments, dim=0),
        #         self.device
        #     ),
        #     move_encoded_inputs_to_device(
        #         unsqueeze_encoded_inputs(baseline1_encoded_segments, dim=0),
        #         self.device
        #     ),
        #     move_encoded_inputs_to_device(
        #         unsqueeze_encoded_inputs(baseline2_encoded_segments, dim=0),
        #         self.device
        #     )
        # )
        ################################################################
        
        return (
            text1_segment_embeddings.unsqueeze(0).to(self.device),
            text2_segment_embeddings.unsqueeze(0).to(self.device),
            baseline1_segment_embeddings.unsqueeze(0).to(self.device),
            baseline2_segment_embeddings.unsqueeze(0).to(self.device)
        )

    def _get_document_embedding(self, segment_embeddings):
        return self.model.get_document_embeddings(
            segment_embeddings, [("text", segment_embeddings.size()[1])]
        )

    def _compute_attributions(
        self,
        input_segment_embeddings,
        baseline_segment_embeddings,
        other_text_embedding,
        n_steps=50,
        internal_batch_size=2
    ):
        self.model.zero_grad()

        attribution_engine = LayerIntegratedGradients(
            self._measure_cosine_similarity, self.model.document_embedder
        )

        segment_attributions = self._compute_integrated_gradients(
            attribution_engine,
            input_segment_embeddings,
            baseline_segment_embeddings,
            other_text_embedding,
            n_steps,
            internal_batch_size
        )

        return segment_attributions   

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

    def _measure_cosine_similarity(
        self, input_segment_embeddings, other_text_embedding
    ):
        input_embedding = self._get_document_embedding(
            input_segment_embeddings
        )
        return torch.cosine_similarity(
            input_embedding, other_text_embedding, dim=0
        )

    def _compute_integrated_gradients(
        self,
        attribution_engine,
        input_segment_embeddings,
        baseline_segment_embeddings,
        other_text_embedding,
        n_steps,
        internal_batch_size
    ):
        self._maybe_disable_cudnn()
        # ^ If the `document_embedder` module of the model is
        #   recurrence-based, we must disable CUDNN because computing
        #   gradients for a recurrent layer is not supported when said
        #   layer is in "eval()" mode
        # (https://github.com/pytorch/captum/blob/master/docs/faq.md#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network)
        attributions, convergence_delta = attribution_engine.attribute(
            inputs=input_segment_embeddings,
            baselines=baseline_segment_embeddings,
            additional_forward_args=(other_text_embedding,),
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=True
        )
        self._maybe_restore_cudnn_enabled_state()

        relative_convergence_delta = torch.abs(
            convergence_delta[0] / torch.sum(attributions)
        )
        if relative_convergence_delta > 0.05:
            print("WARNING: Relative convergence delta is > 5% "
                  f"({relative_convergence_delta*100:.2f}%). You may want to "
                  "recompute attributions passing a larger `n_steps` (current "
                  f"value is {n_steps}).")

        return attributions

    def _maybe_disable_cudnn(self):
        if isinstance(
            self.model.document_embedder, RecurrenceBasedDocumentEmbedder
        ):
            self.was_cudnn_enabled = torch.backends.cudnn.enabled
            torch.backends.cudnn.enabled = False

    def _maybe_restore_cudnn_enabled_state(self):
        if isinstance(
            self.model.document_embedder, RecurrenceBasedDocumentEmbedder
        ):
            torch.backends.cudnn.enabled = self.was_cudnn_enabled
            del self.was_cudnn_enabled
