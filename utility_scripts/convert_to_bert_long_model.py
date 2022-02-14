import shutil

import click

from patents4IPPC.custom_models.bert_long import (
    BertLongConverter, head_type_to_model_class
)


@click.command()
@click.option(
    "-m", "--bert-model", "huggingface_model_name",
    type=str,
    required=True,
    help=("Path to a BERT-based model in HuggingFace format OR name of a "
          "BERT-based model listed in huggingface.co/models.")
)
@click.option(
    "-ht", "--head-type",
    type=click.Choice(list(head_type_to_model_class.keys())),
    default="no-head",
    help=("Type of head that the model to be converted has. This allows "
          "pre-trained weights of that head to be reused. Pass 'no_head' if "
          "the original model has no head or if you want to discard such head "
          "and its associated pre-trained weights.")
)
@click.option(
    "-o", "--output-path",
    type=click.Path(file_okay=False),
    required=True,
    help="Path where the converted model will be saved."
)
@click.option(
    "-w", "--attention-window",
    type=int,
    required=True,
    help="Size of the local attention window."
)
@click.option(
    "-p", "--max-position-embeddings",
    type=int,
    required=True,
    help=("Maximum number of tokens you'd like the model to support after the "
          "conversion.")
)
@click.option(
    "-g", "--global-attention-token", "global_attention_enabled_tokens",
    multiple=True,
    default=["[CLS]"],
    help=("Use this to specify that global attention should be enabled for "
          "this token.")
)
def convert(
    huggingface_model_name,
    head_type,
    output_path,
    attention_window,
    max_position_embeddings,
    global_attention_enabled_tokens
):
    model_class = head_type_to_model_class[head_type]    
    cache_dir = "bert_long_converter_tmp"
    converter = BertLongConverter(
        huggingface_model_name,
        model_class,
        output_path,
        attention_window,
        max_position_embeddings,
        global_attention_enabled_tokens=global_attention_enabled_tokens,
        cache_dir=cache_dir
    )
    converter.convert()
    shutil.rmtree(cache_dir, ignore_errors=True)

if __name__ == "__main__":
    convert() # pylint: disable=no-value-for-parameter
