from pathlib import Path
import copy
import json
import logging

from transformers import (
    AutoTokenizer, BertModel, BertForPreTraining, BertLMHeadModel,
    BertForMaskedLM, BertForNextSentencePrediction,
    BertForSequenceClassification, BertForMultipleChoice,
    BertForTokenClassification, BertForQuestionAnswering,
    LongformerSelfAttention
)
import torch


head_type_to_model_class = {
    "no-head": BertModel,
    "pretraining": BertForPreTraining,
    "language-modeling": BertLMHeadModel,
    "masked-language-modeling": BertForMaskedLM,
    "next-sentence-prediction": BertForNextSentencePrediction,
    "sequence-classification": BertForSequenceClassification,
    "multiple-choice": BertForMultipleChoice,
    "token-classification": BertForTokenClassification,
    "question-answering": BertForQuestionAnswering
}

class BertLongConverter:
    def __init__(
        self,
        huggingface_model_name,
        model_class,
        output_path,
        attention_window=512,
        max_position_embeddings=4096,
        cache_dir=None
    ):
        self.output_path = output_path
        self.attention_window = attention_window
        self.max_position_embeddings = max_position_embeddings

        self.model = model_class.from_pretrained(
            huggingface_model_name, cache_dir=cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            huggingface_model_name,
            model_max_length=self.max_position_embeddings,
            cache_dir=cache_dir
        )
        self.config = self.model.config

        self.current_max_position_embeddings, self.embedding_size = \
            getattr(self.model, "bert", self.model).embeddings.position_embeddings.weight.shape        
        assert self.max_position_embeddings > self.current_max_position_embeddings

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def convert(self):
        extended_position_embeddings = self._extend_position_embeddings()
        self._initialize_and_plug_extended_position_embeddings(
            extended_position_embeddings
        )
        self._replace_attention_layers()
        self._save_converted_model()

        return self.model, self.tokenizer

    def _extend_position_embeddings(self):
        self.tokenizer.model_max_length = self.max_position_embeddings
        self.tokenizer.init_kwargs["model_max_length"] = \
            self.max_position_embeddings
        self.config.max_position_embeddings = self.max_position_embeddings

        extended_position_embeddings = \
            getattr(self.model, "bert", self.model).embeddings.position_embeddings.weight.new_empty(
                self.max_position_embeddings, self.embedding_size
            )
        return extended_position_embeddings

    def _initialize_and_plug_extended_position_embeddings(
        self, extended_position_embeddings
    ):
        # Copy position embeddings over and over to initialize the new
        # position embeddings
        start_idx = 0
        step = self.current_max_position_embeddings
        bert_model = getattr(self.model, "bert", self.model)
        while start_idx < self.max_position_embeddings:
            actual_step = min(self.max_position_embeddings - start_idx, step)
            end_idx = start_idx + actual_step
            extended_position_embeddings[start_idx:end_idx] = \
                bert_model.embeddings.position_embeddings.weight[:actual_step]
            start_idx = end_idx
        bert_model.embeddings.position_embeddings.weight.data = \
            extended_position_embeddings
        bert_model.embeddings.position_ids.data = \
            torch.arange(self.max_position_embeddings).unsqueeze(0)

    def _replace_attention_layers(self):
        self.config.attention_window = self.attention_window
        for i, layer in enumerate(
            getattr(self.model, "bert", self.model).encoder.layer
        ):
            self._replace_single_attention_layer(layer, layer_id=i)      

    def _save_converted_model(self):
        self.logger.info("Saving model to %s", self.output_path)
        self.config._name_or_path = str(self.output_path)
        self.model.save_pretrained(self.output_path)
        self.tokenizer.save_pretrained(self.output_path)
        self._fix_config_files()

    def _replace_single_attention_layer(self, layer, layer_id):
        longformer_self_attention = LongformerSelfAttention(
            self.config, layer_id=layer_id
        )
        longformer_self_attention.query = layer.attention.self.query
        longformer_self_attention.key = layer.attention.self.key
        longformer_self_attention.value = layer.attention.self.value

        # TODO: Check if it's possible to avoid allocating these tensors,
        # assuming we don't need global attention.
        longformer_self_attention.query_global = copy.deepcopy(
            layer.attention.self.query
        )
        longformer_self_attention.key_global = copy.deepcopy(
            layer.attention.self.key
        )
        longformer_self_attention.value_global = copy.deepcopy(
            layer.attention.self.value
        )

        layer.attention.self = longformer_self_attention          

    def _fix_config_files(self):
        self._fix_model_config_file()
        self._fix_tokenizer_config_file()

    def _fix_model_config_file(self):
        config_file = Path(self.output_path) / "config.json"
        saved_config = json.loads(config_file.read_text())
        saved_config["architectures"] = list(map(
            lambda a: a.replace("Bert", "BertLong"),
            saved_config["architectures"]
        ))
        saved_config["model_type"] = "bert-long"
        config_file.write_text(
            json.dumps(saved_config, indent=2, sort_keys=True) + "\n"
        )        

    def _fix_tokenizer_config_file(self):
        config_file = Path(self.output_path) / "tokenizer_config.json"
        saved_config = json.loads(config_file.read_text())
        saved_config["name_or_path"] = str(self.output_path)
        saved_config["tokenizer_class"] = \
            saved_config["tokenizer_class"].replace("Bert", "BertLong")
        config_file.write_text(
            json.dumps(saved_config, indent=2, sort_keys=True) + "\n"
        )
