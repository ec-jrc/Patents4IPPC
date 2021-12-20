from transformers import BertTokenizer, BertTokenizerFast


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

    def __call__(self, *args, **kwargs):
        return _pad_to_multiple_of_attention_window(self, *args, **kwargs)


class BertLongTokenizerFast(BertTokenizerFast):
    slow_tokenizer_class = BertLongTokenizer

    def __init__(self, *args, attention_window=512, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_window = attention_window

    def __call__(self, *args, **kwargs):
        return _pad_to_multiple_of_attention_window(self, *args, **kwargs)
