from transformers import (
    AutoConfig, AutoModel, AutoModelForPreTraining, AutoModelForCausalLM,
    AutoModelForMaskedLM, AutoModelForNextSentencePrediction,
    AutoModelForSequenceClassification, AutoModelForMultipleChoice,
    AutoModelForTokenClassification, AutoModelForQuestionAnswering,
    AutoTokenizer
)

from .custom_models.bert_long import (
    BertLongConfig, BertLongModel, BertLongForPreTraining, BertLongLMHeadModel,
    BertLongForMaskedLM, BertLongForNextSentencePrediction,
    BertLongForSequenceClassification, BertLongForMultipleChoice,
    BertLongForTokenClassification, BertLongForQuestionAnswering,
    BertLongTokenizer, BertLongTokenizerFast
)

# pylint: disable=no-member
AutoConfig.register("bert-long", BertLongConfig)

AutoModel.register(BertLongConfig, BertLongModel)
AutoModelForPreTraining.register(BertLongConfig, BertLongForPreTraining)
AutoModelForCausalLM.register(BertLongConfig, BertLongLMHeadModel)
AutoModelForMaskedLM.register(BertLongConfig, BertLongForMaskedLM)
AutoModelForNextSentencePrediction.register(
    BertLongConfig, BertLongForNextSentencePrediction
)
AutoModelForSequenceClassification.register(
    BertLongConfig, BertLongForSequenceClassification
)
AutoModelForMultipleChoice.register(BertLongConfig, BertLongForMultipleChoice)
AutoModelForTokenClassification.register(
    BertLongConfig, BertLongForTokenClassification
)
AutoModelForQuestionAnswering.register(
    BertLongConfig, BertLongForQuestionAnswering
)

AutoTokenizer.register(
    BertLongConfig,
    slow_tokenizer_class=BertLongTokenizer,
    fast_tokenizer_class=BertLongTokenizerFast
)
# pylint: enable=no-member
