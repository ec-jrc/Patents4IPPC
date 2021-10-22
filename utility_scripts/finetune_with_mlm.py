from pathlib import Path
import json

from sklearn.model_selection import train_test_split
import pandas as pd
import click

from patents4IPPC.finetuning import mlm_adaptive_tuning


@click.command(help=('Fine-tune a HuggingFace transformers model on a Masked '
                     'Language Modeling (MLM) task.'))
@click.option(
    '-m', '--model', 'path_to_model_dir',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help=('Path to a pre-trained HuggingFace transformers model to further '
          'fine-tune.')
)
@click.option(
    '-d', '--dataset', 'path_to_dataset',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help='Path to a dataset to use for MLM fine-tuning.'
)
@click.option(
    '-dc', '--dataset-column',
    required=True,
    help='Column of the dataset to use for MLM fine-tuning.'
)
@click.option(
    '-vl', '--validation-portion',
    type=float,
    required=True,
    help='Fraction of the dataset to use for validation.'
)
@click.option(
    '--split-seed',
    type=int,
    default=None,
    help='Random seed to use when splitting the dataset into train/validation.'
)
@click.option(
    '--train-seed',
    type=int,
    default=None,
    help='Random seed to use when training the model.'
)
@click.option(
    '-c', '--config-file', 'path_to_config_file',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help=('Path to a JSON file containing the hyperparameters to use when '
          'fine-tuning the model.')
)
@click.option(
    '-o', '--output-dir',
    type=click.Path(file_okay=False),
    required=True,
    help='Directory where the fine-tuned model will be saved.'
)
def main(
    path_to_model_dir,
    path_to_dataset,
    dataset_column,
    validation_portion,
    split_seed,
    train_seed,
    path_to_config_file,
    output_dir
):
    # Load the dataset
    dataset = pd.read_csv(path_to_dataset)[dataset_column]
    dataset_train, dataset_valid = train_test_split(
        dataset,
        test_size=validation_portion,
        shuffle=True,
        random_state=split_seed
    )

    # Load the configuration file
    config = json.loads(Path(path_to_config_file).read_text())
    
    # Start the MLM fine-tuning
    mlm_adaptive_tuning.mlm_finetuning(
        model_name_or_path=path_to_model_dir,
        train_texts=dataset_train.values.tolist(),
        dev_texts=dataset_valid.values.tolist(),
        output_dir=output_dir,
        seed=train_seed,
        **config
    )

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
