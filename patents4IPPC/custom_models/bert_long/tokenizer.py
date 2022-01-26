from transformers import AutoConfig, BertTokenizer, BertTokenizerFast


def _pad_to_multiple_of_attention_window(self, *args, **kwargs):
    # `LongformerSelfAttention` expects its input to have a length
    # that is a multiple of `attention_window`, so we must pad it here
    kwargs.pop('padding', None)
    kwargs.pop('pad_to_multiple_of', None)
    return super(self.__class__, self).__call__(
        *args,
        padding=True,
        pad_to_multiple_of=self.attention_window,
        **kwargs
    )
    # NOTE: This function returns, among other things, an
    # `attention_mask` tensor made of 0s and 1s, where 0 = no attention
    # and 1 = local (windowed) attention. If you want to add global
    # attention, you should make sure the corresponding values in
    # `attention_mask` are set to 2.

class BertLongTokenizer(BertTokenizer):
    def __init__(self, *args, attention_window=512, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_window = attention_window

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *init_inputs, **kwargs
    ):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        assert hasattr(config, "attention_window"), \
            "No value for `attention_window` was found in the configuration file."
        assert min(config.attention_window) == max(config.attention_window), \
            "Different `attention_window` values for different layers is not supported."

        return super(BertLongTokenizer, cls).from_pretrained(
            pretrained_model_name_or_path,
            *init_inputs,
            attention_window=config.attention_window[0],
            **kwargs
        )

    def __call__(self, *args, **kwargs):
        return _pad_to_multiple_of_attention_window(self, *args, **kwargs)


class BertLongTokenizerFast(BertTokenizerFast):
    slow_tokenizer_class = BertLongTokenizer

    def __init__(self, *args, attention_window=512, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_window = attention_window

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *init_inputs, **kwargs
    ):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        assert hasattr(config, "attention_window"), \
            "No value for `attention_window` was found in the configuration file."
        assert min(config.attention_window) == max(config.attention_window), \
            "Different `attention_window` values for different layers is not supported."

        return super(BertLongTokenizerFast, cls).from_pretrained(
            pretrained_model_name_or_path,
            *init_inputs,
            attention_window=config.attention_window[0],
            **kwargs
        )

    def __call__(self, *args, **kwargs):
        return _pad_to_multiple_of_attention_window(self, *args, **kwargs)
