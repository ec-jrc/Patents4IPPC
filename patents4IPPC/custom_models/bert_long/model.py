from transformers import (
    BertModel, BertForPreTraining, BertLMHeadModel, BertForMaskedLM,
    BertForNextSentencePrediction, BertForSequenceClassification,
    BertForMultipleChoice, BertForTokenClassification,
    BertForQuestionAnswering, LongformerSelfAttention
)

from .config import BertLongConfig


class BertLongSelfAttention(LongformerSelfAttention):
    def forward(self, *args, **kwargs):
        #https://github.com/huggingface/transformers/issues/9588#issuecomment-763487508
        hidden_states, attention_mask, layer_head_mask = args[:3]
        attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = any(is_index_global_attn.flatten())
        output_attentions = args[-1]

        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions
        )

def _replace_self_attention_layers(self, config):
    for i, layer in enumerate(getattr(self, "bert", self).encoder.layer):
        layer.attention.self = BertLongSelfAttention(config, layer_id=i)

class BertLongModel(BertModel):
    config_class = BertLongConfig
    def __init__(self, config):
        super().__init__(config)
        _replace_self_attention_layers(self, config)

class BertLongForPreTraining(BertForPreTraining):
    config_class = BertLongConfig
    def __init__(self, config):
        super().__init__(config)
        _replace_self_attention_layers(self, config)

class BertLongLMHeadModel(BertLMHeadModel):
    config_class = BertLongConfig
    def __init__(self, config):
        super().__init__(config)
        _replace_self_attention_layers(self, config)

class BertLongForMaskedLM(BertForMaskedLM):
    config_class = BertLongConfig
    def __init__(self, config):
        super().__init__(config)
        _replace_self_attention_layers(self, config)

class BertLongForNextSentencePrediction(BertForNextSentencePrediction):
    config_class = BertLongConfig
    def __init__(self, config):
        super().__init__(config)
        _replace_self_attention_layers(self, config)

class BertLongForSequenceClassification(BertForSequenceClassification):
    config_class = BertLongConfig
    def __init__(self, config):
        super().__init__(config)
        _replace_self_attention_layers(self, config)

class BertLongForMultipleChoice(BertForMultipleChoice):
    config_class = BertLongConfig
    def __init__(self, config):
        super().__init__(config)
        _replace_self_attention_layers(self, config)

class BertLongForTokenClassification(BertForTokenClassification):
    config_class = BertLongConfig
    def __init__(self, config):
        super().__init__(config)
        _replace_self_attention_layers(self, config)

class BertLongForQuestionAnswering(BertForQuestionAnswering):
    config_class = BertLongConfig
    def __init__(self, config):
        super().__init__(config)
        _replace_self_attention_layers(self, config)
