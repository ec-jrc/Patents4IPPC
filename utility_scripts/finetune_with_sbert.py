from pathlib import Path
import json

import pandas as pd
import click

from patents4IPPC.finetuning import sbert_finetuning
import utils


@click.command(help=('Fine-tune a HuggingFace transformers or '
                     'sentence-transformers model on a classification or '
                     'Semantic Textual Similarity (STS) task.'))
@click.option(
    '-m', '--model', 'path_to_model_dir',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help=('Path to a pre-trained HuggingFace transformers or '
          'sentence-transformers model to further fine-tune.')
)
@click.option(
    '-s', '--sbert-model', 'is_sbert_model',
    is_flag=True,
    help=('Used to indicate that the pre-trained model is a '
          '`sentence-transformers` checkpoint and not a plain '
          'HuggingFace transformers model.')
)
@click.option(
    '-d', '--dataset', 'path_to_dataset',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help=('Path to a dataset with "standard" column names, i.e. it must have '
          'at least the following columns: query_id, query, response_id, '
          'response, label.')
)
@click.option(
    '-vl', '--validation-portion',
    type=float,
    default=None,
    help=('Fraction of the dataset to use for validation. Required unless the '
          'dataset has pre-determined train/dev/test splits (like NLI) or a '
          'separate validation dataset was provided via the "--dev-dataset" '
          'option.')
)
@click.option(
    '--split-seed',
    type=int,
    default=None,
    help=('Random seed to use when splitting the dataset into '
          'train/validation. Ignored if "--dev-dataset" was specified.')
)
@click.option(
    '-dd', '--dev-dataset', 'path_to_dev_dataset',
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=('Path to a dataset to be used for validation purposes (in case you '
          'do not intend to use part of the dataset specified via the '
          '"--dataset" option).')    
)
@click.option(
    '-l', '--loss',
    type=click.Choice(['softmax', 'cosine']),
    help=('Loss function to minimize. Use "softmax" for classification tasks '
          '(e.g. NLI) and "cosine" for semantic textual similarity tasks.')
)
@click.option(
    '-c', '--config-file', 'path_to_config_file',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help=('Path to a JSON file containing the hyperparameters to use when '
          'fine-tuning the model.')
)
@click.option(
    '-n', '--normalize-labels',
    is_flag=True,
    help='Normalize dataset labels in the [0,1] range.'
)
@click.option(
    '-o', '--output-path',
    type=click.Path(file_okay=False),
    required=True,
    help='Path where the fine-tuned model will be saved.'
)
def main(
    path_to_model_dir,
    is_sbert_model,
    path_to_dataset,
    validation_portion,
    split_seed,
    path_to_dev_dataset,
    loss,
    path_to_config_file,
    normalize_labels,
    output_path,
):
    assert path_to_dev_dataset is not None or validation_portion is not None, \
           'Please specify what fraction of the dataset to use for validation.'
    
    # Load the dataset
    valid_portion = validation_portion if path_to_dev_dataset is None else 0.0
    train_portion = 1 - valid_portion    
    dataset_train, dataset_valid, _ = utils.load_dataset(
        path_to_dataset,
        train_portion=train_portion,
        valid_portion=valid_portion,
        test_portion=0.0,
        seed=split_seed,
        normalize_labels=normalize_labels
    )

    # Start the fine-tuning
    if path_to_dev_dataset is not None:
        # Load the validation dataset
        dev_dataset_train, dev_dataset_valid, dev_dataset_test = \
            utils.load_dataset(
                path_to_dev_dataset,
                train_portion=1.0,
                valid_portion=0.0,
                test_portion=0.0,
                seed=split_seed
            )
        # The code below might look like it makes no sense - why call
        # pd.concat() when I've already specified train_portion=1.0,
        # valid_portion=0.0 and test_portion=0.0? -, although it's
        # actually intended to work with "special" datasets like STSb
        # that have predefined train/dev/test splits and thus disregard
        # the values of train_portion, valid_portion and test_portion            
        dataset_valid = pd.concat(
            [dev_dataset_train, dev_dataset_valid, dev_dataset_test],
            axis=0,
            ignore_index=True
        )

    train_samples = list(
        dataset_train[['query', 'response', 'label']].itertuples(
            index=False, name=None
        )
    )
    valid_samples = list(
        dataset_valid[['query', 'response', 'label']].itertuples(
            index=False, name=None
        )
    )

    config = json.loads(Path(path_to_config_file).read_text())
    sbert_finetuning.sentence_transformers_finetuning(
        model_name_or_path=path_to_model_dir,
        train_samples=train_samples,
        dev_samples=valid_samples,
        loss=loss,
        output_path=output_path,
        is_sbert_model=is_sbert_model,
        **config
    )

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
