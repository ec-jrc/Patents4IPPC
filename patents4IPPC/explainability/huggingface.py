from copy import deepcopy

from captum.attr import LayerIntegratedGradients
from transformers import AutoModel, AutoTokenizer
import torch

import utils


class HuggingFaceTextSimilarityExplainer:
    
    def __init__(self, model_name_or_path, pooling_mode="mean"):
        self.pooling_mode = pooling_mode

        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model.to(self.device)

    def explain(
        self,
        text1,
        text2,
        n_steps=50,
        internal_batch_size=2,
        normalize_attributions=False
    ):
        text1_encoding, text2_encoding, baseline1_encoding, baseline2_encoding = \
            self._construct_baselines_and_inputs(text1, text2)
        
        # Explain text1 while keeping text2 freezed
        text2_embedding = self._embed_text(**text2_encoding)
        text2_embedding.detach()
        text1_attributions = self._compute_attributions(
            text1_encoding,
            baseline1_encoding,
            text2_embedding,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size
        )

        # Explain text2 while keeping text1 freezed
        text1_embedding = self._embed_text(**text1_encoding)
        text1_embedding.detach()
        text2_attributions = self._compute_attributions(
            text2_encoding,
            baseline2_encoding,
            text1_embedding,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size
        )

        # Summarize and optionally normalize attributions
        text1_attributions = self._summarize_attributions(text1_attributions)
        text2_attributions = self._summarize_attributions(text2_attributions)
        if normalize_attributions:
            text1_attributions = self._normalize_attributions(text1_attributions)
            text2_attributions = self._normalize_attributions(text2_attributions)

        text1_tokens = self.tokenizer.convert_ids_to_tokens(
            text1_encoding["input_ids"].squeeze(0).detach().cpu().tolist()
        )
        text2_tokens = self.tokenizer.convert_ids_to_tokens(
            text2_encoding["input_ids"].squeeze(0).detach().cpu().tolist()
        )

        return (
            text1_tokens, text1_attributions, text2_tokens, text2_attributions
        )

    def _construct_baselines_and_inputs(self, text1, text2):
        text1_encoding = self.tokenizer(text1)
        text2_encoding = self.tokenizer(text2)

        baseline1_encoding = deepcopy(text1_encoding)
        baseline1_encoding["input_ids"] = [
            token_id
            if token_id in self.tokenizer.all_special_ids
            else self.tokenizer.pad_token_id
            for token_id in baseline1_encoding["input_ids"]
        ]
        
        baseline2_encoding = deepcopy(text2_encoding)
        baseline2_encoding["input_ids"] = [
            token_id
            if token_id in self.tokenizer.all_special_ids
            else self.tokenizer.pad_token_id
            for token_id in baseline2_encoding["input_ids"]
        ]
        
        return (
            (text1_encoding
             .convert_to_tensors("pt", prepend_batch_axis=True)
             .to(device=self.device)),
            (text2_encoding
             .convert_to_tensors("pt", prepend_batch_axis=True)
             .to(device=self.device)),
            (baseline1_encoding
             .convert_to_tensors("pt", prepend_batch_axis=True)
             .to(device=self.device)),
            (baseline2_encoding
             .convert_to_tensors("pt", prepend_batch_axis=True)
             .to(device=self.device))
        )

    def _embed_text(self, input_ids, attention_mask, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return utils.pool_embeddings_with_attention_mask(
            embeddings=outputs.last_hidden_state,
            attention_mask=attention_mask,
            mode=self.pooling_mode
        )

    def _compute_attributions(
        self,
        input_encoding,
        baseline_encoding,
        other_text_embedding,
        n_steps=50,
        internal_batch_size=2
    ):
        self.model.zero_grad()

        attribution_engine = LayerIntegratedGradients(
            self._measure_cosine_similarity,
            self.model.embeddings.word_embeddings
        )

        attributions, convergence_delta = attribution_engine.attribute(
            inputs=input_encoding["input_ids"],
            baselines=baseline_encoding["input_ids"],
            additional_forward_args=(
                input_encoding["attention_mask"], other_text_embedding
            ),
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

    def _summarize_attributions(self, attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions

    def _normalize_attributions(self, attributions):
        signs = torch.sign(attributions)
        attribution_magnitudes = torch.abs(attributions)
        normalized_attribution_magnitudes = (attribution_magnitudes
                                             / attribution_magnitudes.max())
        return signs * normalized_attribution_magnitudes   

    def _measure_cosine_similarity(
        self, input_ids, attention_mask, other_text_embedding
    ):
        input_embedding = self._embed_text(input_ids, attention_mask)
        return torch.cosine_similarity(
            input_embedding, other_text_embedding, dim=1
        )
