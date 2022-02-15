from pathlib import Path

import click
from transformers import AutoModel, AutoTokenizer

from patents4IPPC.custom_models.hierarchical_transformer import (
    DocumentSimilarityDataset,
    DocumentSimilarityTrainer,
    HierarchicalTransformer,
    TrainingArguments
)
from patents4IPPC.custom_models.hierarchical_transformer.embedding_documents \
    import (
        DocumentEmbedderType,
        RecurrenceBasedDocumentEmbedderConfig,
        TransformerBasedDocumentEmbedderConfig
    )


def create_new_model(
    path_to_segment_transformer,
    document_embedder_type,
    path_to_document_embedder_config,
    segment_transformer_inner_batch_size,
    segment_transformer_pooling_mode
):
    segment_transformer = AutoModel.from_pretrained(path_to_segment_transformer)
    tokenizer = AutoTokenizer.from_pretrained(path_to_segment_transformer)

    embedder_type_to_config_class = {
        "recurrence": RecurrenceBasedDocumentEmbedderConfig,
        "transformer": TransformerBasedDocumentEmbedderConfig
    }
    document_embedder_config_class = embedder_type_to_config_class[
        document_embedder_type
    ]
    document_embedder_config = document_embedder_config_class.from_json(
        path_to_document_embedder_config
    )

    embedder_type_to_enum_value = {
        "recurrence": DocumentEmbedderType.RECURRENCE,
        "transformer": DocumentEmbedderType.TRANSFORMER
    }
    model = HierarchicalTransformer(
        segment_transformer,
        embedder_type_to_enum_value[document_embedder_type],
        document_embedder_config,
        segment_transformer_inner_batch_size,
        segment_transformer_pooling_mode
    )

    return model, tokenizer

def load_pretrained_model_and_tokenizer(
    path_to_pretrained_model_dir, segment_transformer_inner_batch_size
):
    model = HierarchicalTransformer.from_pretrained(
        path_to_pretrained_model_dir, segment_transformer_inner_batch_size
    )
    segment_transformer_dir = \
        Path(path_to_pretrained_model_dir) / "segment_transformer"
    tokenizer = AutoTokenizer.from_pretrained(str(segment_transformer_dir))

    return model, tokenizer


@click.command(
    help="Fine-tune a HierarchicalTransformer model on a Text Similarity task."
)
@click.option(
    "-m", "--pretrained-model", "path_to_pretrained_model_dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help=("Path to a pre-trained HierarchicalTransformer model. Use this "
          "argument to provide the checkpoint of a HierarchicalTransformer "
          "model that was previously fine-tuned using this same script.")
)
@click.option(
    "-s", "--segment-transformer", "path_to_segment_transformer",
    type=click.Path(file_okay=False),
    default=None,
    help=("Path to a pre-trained HuggingFace transformers model to be used to "
          "encode segments. Ignored if \"--pretrained-model\" was specified.")
)
@click.option(
    "-b", "--segment-transformer-inner-batch-size",
    type=int,
    required=True,
    help=("Inner batch size of the segment transformer. NOTE: this is NOT its "
          "effective batch size, but rather the amount of samples on which it "
          "can perform a forward pass without incurring in an OOM error. The "
          "actual batch size depends on the total number of segments within a "
          "batch of documents. For higher efficiency, set this to the maximum "
          "value that the model can handle.")
)
@click.option(
    "-p", "--segment-transformer-pooling-mode",
    type=click.Choice(["cls", "max", "mean"]),
    default="mean",    
    help=("Pooling strategy to use in the segment transformer to go from "
          "token embeddings to segment embeddings. Ignored if "
          "'--pretrained-model' was specified.")
)
@click.option(
    "-t", "--document-embedder-type",
    type=click.Choice(["recurrence", "transformer"]),
    default="transformer",    
    help=("Type of model to use to obtain document embeddings from segment "
          "embeddings. Ignored if \"--pretrained-model\" was specified.")
)
@click.option(
    "-ea", "--encoder-attribute-name", "segment_transformer_encoder_attr_name",
    type=str,
    default="encoder",
    help=("Within the segment Transformer, name of the attribute that holds "
          "the encoder module of the Transformer. Needed in case you want to "
          "train some of the top layers of the model.")
)
@click.option(
    "-td", "--train-dataset-dir", "path_to_train_dataset_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help=("Path to a directory containing a dataset for document similarity "
          "to be used for training.")
)
@click.option(
    "-ed", "--eval-dataset-dir", "path_to_eval_dataset_dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help=("Path to a directory containing a dataset for document similarity "
          "to be used for evaluation purposes.")
)
@click.option(
    "-c", "--config-file", "path_to_document_embedder_config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help=("Path to a JSON configuration file containing the hyperparameters "
          "to use for the document embedder when finetuning the model. "
          "Ignored if \"--pretrained-model\" was specified.")
)
@click.option(
    "-a", "--training-arguments", "path_to_training_arguments",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to a JSON file containing the training arguments."
)
@click.option(
    "-e", "--epochs",
    type=int,
    required=True,
    help="Number of training epochs."
)
@click.option(
    "-o", "--output-dir",
    type=click.Path(file_okay=False),
    required=True,
    help="Directory where the fine-tuned model will be saved."
)
@click.option(
    "-cd", "--cache-dir",
    type=click.Path(file_okay=False),
    default="cached_segment_embeddings",
    help=("Directory where segment embeddings will be cached, if necessary. "
          "In particular, they will be cached only if you decided to freeze "
          "all of the segment Transformer's weights.")
)
def main(
    path_to_pretrained_model_dir,
    path_to_segment_transformer,
    segment_transformer_inner_batch_size,
    segment_transformer_pooling_mode,
    document_embedder_type,
    segment_transformer_encoder_attr_name,
    path_to_train_dataset_dir,
    path_to_eval_dataset_dir,
    path_to_document_embedder_config,
    path_to_training_arguments,
    epochs,
    output_dir,
    cache_dir
):
    if path_to_pretrained_model_dir is not None:
        model, tokenizer = load_pretrained_model_and_tokenizer(
            path_to_pretrained_model_dir, segment_transformer_inner_batch_size
        )
    else:
        model, tokenizer = create_new_model(
            path_to_segment_transformer,
            document_embedder_type,
            path_to_document_embedder_config,
            segment_transformer_inner_batch_size,
            segment_transformer_pooling_mode
        )

    train_dataset = DocumentSimilarityDataset.from_directory(
        path_to_train_dataset_dir,
        tokenizer,
        model.segment_transformer.config.max_position_embeddings
    )
    eval_dataset = DocumentSimilarityDataset.from_directory(
        path_to_eval_dataset_dir,
        tokenizer,
        model.segment_transformer.config.max_position_embeddings
    )

    training_arguments = TrainingArguments.from_json(
        path_to_training_arguments
    )
    trainer = DocumentSimilarityTrainer(
        model,
        segment_transformer_encoder_attr_name=segment_transformer_encoder_attr_name,
        train_dataset=train_dataset,
        training_arguments=training_arguments,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        cache_dir=cache_dir
    )
    trainer.train(num_epochs=epochs)


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
