from transformers import BertConfig


class BertLongConfig(BertConfig):
    model_type = "bert-long"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
