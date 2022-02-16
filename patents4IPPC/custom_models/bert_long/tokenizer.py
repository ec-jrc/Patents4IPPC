from transformers import AutoConfig, BertTokenizer, BertTokenizerFast
import numpy as np
import torch


def _pad_to_multiple_of_attention_window(self, *args, **kwargs):
    # `LongformerSelfAttention` expects its input to have a length
    # that is a multiple of `attention_window`, so we must pad it here
    kwargs.pop('padding', None)
    kwargs.pop('pad_to_multiple_of', None)
    return super(self.__class__, self).__call__(
        *args,
        padding=True,
        pad_to_multiple_of=self.model_attention_window,
        **kwargs
    )

def _mark_tokens_needing_global_attention(self, encoded_inputs):
    GLOBAL_ATTENTION_MARKER = 2
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]
    if isinstance(attention_mask, torch.Tensor):
        for t in self.global_attention_enabled_tokens:
            attention_mask[input_ids == t] = GLOBAL_ATTENTION_MARKER
    elif isinstance(attention_mask, list):
        input_ids_arr = np.array(input_ids)
        attention_mask_arr = np.array(attention_mask)
        for t in self.global_attention_enabled_tokens:
            attention_mask_arr[input_ids_arr == t] = GLOBAL_ATTENTION_MARKER
        encoded_inputs["attention_mask"] = attention_mask_arr.tolist()
    else:
        raise ValueError(
            f"Unsupported type '{type(attention_mask)}' for `attention_mask.`"
        )
    
    return encoded_inputs

class BertLongTokenizer(BertTokenizer):
    def __init__(
        self,
        *args,
        model_attention_window=512,
        global_attention_enabled_tokens=None,
        **kwargs
    ):
        self.model_attention_window = model_attention_window
        self.global_attention_enabled_tokens = \
            global_attention_enabled_tokens or ["[CLS]"]
        super().__init__(
            *args,
            global_attention_enabled_tokens=global_attention_enabled_tokens,
            # ^ Pass this to `BertTokenizer` so that it'll be saved in 
            #   the "tokenizer_config.json" file 
            **kwargs
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *init_inputs, **kwargs
    ):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        assert hasattr(config, "attention_window"), \
            ("No value for `attention_window` was found in the model's "
             "configuration file.")
        assert min(config.attention_window) == max(config.attention_window), \
            ("Different `attention_window` values for different layers "
             "of the model is not supported.")

        return super(BertLongTokenizer, cls).from_pretrained(
            pretrained_model_name_or_path,
            *init_inputs,
            model_attention_window=config.attention_window[0],
            **kwargs
        )

    def __call__(self, *args, **kwargs):
        encoded_inputs = _pad_to_multiple_of_attention_window(
            self, *args, **kwargs
        )
        return _mark_tokens_needing_global_attention(self, encoded_inputs)


class BertLongTokenizerFast(BertTokenizerFast):
    slow_tokenizer_class = BertLongTokenizer

    def __init__(
        self,
        *args,
        model_attention_window=512,
        global_attention_enabled_tokens=None,
        **kwargs
    ):
        self.model_attention_window = model_attention_window
        self.global_attention_enabled_tokens = \
            global_attention_enabled_tokens or ["[CLS]"]
        super().__init__(
            *args,
            global_attention_enabled_tokens=global_attention_enabled_tokens,
            # ^ Pass this to `BertTokenizer` so that it'll be saved in 
            #   the "tokenizer_config.json" file 
            **kwargs
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *init_inputs, **kwargs
    ):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        assert hasattr(config, "attention_window"), \
            ("No value for `attention_window` was found in the model's "
             "configuration file.")
        assert min(config.attention_window) == max(config.attention_window), \
            ("Different `attention_window` values for different layers "
             "of the model is not supported.")

        return super(BertLongTokenizerFast, cls).from_pretrained(
            pretrained_model_name_or_path,
            *init_inputs,
            model_attention_window=config.attention_window[0],
            **kwargs
        )

    def __call__(self, *args, **kwargs):
        encoded_inputs = _pad_to_multiple_of_attention_window(
            self, *args, **kwargs
        )
        return _mark_tokens_needing_global_attention(self, encoded_inputs)
