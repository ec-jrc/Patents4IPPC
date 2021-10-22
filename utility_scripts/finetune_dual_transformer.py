from pathlib import Path
import json

import pandas as pd
import joblib
import torch
import click

from patents4IPPC.custom_models.dual_transformer import (
    RankingDataset, DualTransformer, NeuralNDCGLoss, train_dual_transformer
)
import utils


@click.command(help='Fine-tune a DualTransformer model on a Text Ranking task.')
@click.option(
    '-m', '--pretrained-model', 'path_to_pretrained_model_dir',
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help=('Path to a pre-trained DualTransformer model. Use this argument to '
          'provide the checkpoint of a DualTransformer model that was '
          'previously fine-tuned using this same script.')
)
@click.option(
    '-qm', '--query-model', 'path_to_query_model_dir',
    type=click.Path(file_okay=False),
    default=None,
    help=('Path to a pre-trained HuggingFace transformers model to be used to '
          'encode queries. Ignored if "--pretrained-model" was specified.')
)
@click.option(
    '-rm', '--response-model', 'path_to_response_model_dir',
    type=click.Path(file_okay=False),
    default=None, # If None, it'll default to the query model
    help=('Path to a pre-trained HuggingFace transformers model to be used to '
          'encode responses. If not specified, it will default to the query '
          'model. Ignored if "--pretrained-model" was specified.')
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
    '-e', '--response-embeddings', 'path_to_response_embeddings',
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=('Path to a pickled dictionary containing pre-computed embeddings '
          'for the responses in the dataset.')
)
@click.option(
    '-iqm', '--include-query-mapper-regardless',
    is_flag=True,
    help=('Add a query mapper on top of the query embedder regardless of '
          'whether the query and response embedders have different embedding '
          'sizes or not.')
)
@click.option(
    '-f', '--freeze-embedders-weights',
    is_flag=True,
    help=('Do not train embedders weights, but just the query mapper. If this '
          'flag is specified, "--include-query-mapper-regardless" must also '
          'be specified unless query and response embedders have different '
          'embedding sizes (in which case the query mapper is added '
          'automatically).')
)
@click.option(
    '-c', '--config-file', 'path_to_config_file',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help=('Path to a JSON file containing the hyperparameters to use when '
          'finetuning the model.')
)
@click.option(
    '-et', '--run-evaluation-on-training-set',
    is_flag=True,
    help=('Run the evaluation on the training set after each epoch of '
          'training. May slow down training as it takes as much as a training '
          'epoch, which means that enabling this roughly makes the training '
          'twice as slow as it normally would be.')
)
@click.option(
    '-i', '--iterative-mode',
    is_flag=True,
    help=('Process each batch of training samples in an iterative fashion '
          'rather than in a vectorized fashion. Helps saving memory at the '
          'expense of a small overhead in computation.')
)
@click.option(
    '-o', '--output-dir', 'path_to_output_dir',
    type=click.Path(file_okay=False),
    required=True,
    help='Directory where the fine-tuned model will be saved.'
)
def main(
    path_to_pretrained_model_dir,
    path_to_query_model_dir,
    path_to_response_model_dir,
    path_to_dataset,
    validation_portion,
    split_seed,
    path_to_dev_dataset,
    path_to_response_embeddings,
    include_query_mapper_regardless,
    freeze_embedders_weights,
    path_to_config_file,
    run_evaluation_on_training_set,
    iterative_mode,
    path_to_output_dir
):
    assert (path_to_response_model_dir is None
            or path_to_response_embeddings is None), \
           ('You have provided both a response model AND pre-computed '
            'response embeddings, which is not supported.')

    assert path_to_dev_dataset is not None or validation_portion is not None, \
           'Please specify what fraction of the dataset to use for validation.'

    # TODO: Pre-computed response embeddings do not go well with a
    # dedicated validation dataset (i.e. different than the one used for
    # training). Either remove that option or add another one for
    # providing pre-computed embeddings for the responses contained in
    # the dedicated validation dataset too.

    # Load the config file
    config = json.loads(Path(path_to_config_file).read_text())

    # Optionally load the pickled dictionary containing precomputed
    # response embeddings
    if path_to_response_embeddings is not None:
        precomputed_response_embeddings = joblib.load(
            path_to_response_embeddings
        )
        precomputed_response_emb_size = \
            len(list(precomputed_response_embeddings.values())[0])
    else:
        precomputed_response_embeddings = None 
        precomputed_response_emb_size = None

    # Create the model
    if path_to_pretrained_model_dir is not None:
        model = DualTransformer.from_pretrained(
            path_to_pretrained_model_dir,
            freeze_embedders_weights=freeze_embedders_weights
        )
    else:
        model = DualTransformer(
            path_to_pretrained_query_embedder=path_to_query_model_dir,
            path_to_pretrained_response_embedder=path_to_response_model_dir,
            max_sentence_length=config.get('max_sequence_length', None),
            precomputed_response_embeddings_size=precomputed_response_emb_size,
            freeze_embedders_weights=freeze_embedders_weights,
            include_query_mapper_regardless=include_query_mapper_regardless,
            query_mapper_hidden_size=config.get('query_mapper_hidden_size', 2048)
        )
    model.to_device('cuda')

    # Load the dataset
    valid_portion = validation_portion if path_to_dev_dataset is None else 0.0
    train_portion = 1 - valid_portion
    dataset_train, dataset_valid, _ = utils.load_dataset(
        path_to_dataset,
        train_portion=train_portion,
        valid_portion=valid_portion,
        test_portion=0.0,
        seed=split_seed,
        normalize_labels=True
    )
    if path_to_dev_dataset is not None:
        # Load the validation dataset
        dev_dataset_train, dev_dataset_valid, dev_dataset_test = \
            utils.load_dataset(
                path_to_dev_dataset,
                train_portion=1.0,
                valid_portion=0.0,
                test_portion=0.0,
                seed=split_seed,
                normalize_labels=True
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

    # Setup the dataloaders
    train_dataset = RankingDataset(
        dataset_train,
        min_list_length=config['min_list_length'],
        precomputed_embeddings_mode=precomputed_response_embeddings is not None
    )
    valid_dataset = RankingDataset(
        dataset_valid,
        min_list_length=config['min_list_length'],
        precomputed_embeddings_mode=precomputed_response_embeddings is not None
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config['batch_size'], shuffle=True
    )

    # Define loss and optimizer
    loss_fn = NeuralNDCGLoss(config['cosine_loss_weight'])
    optimizer = torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Start the finetuning
    train_loss_history, eval_loss_history = train_dual_transformer(
        dual_transformer_model=model,
        train_dataloader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=config['epochs'],
        eval_dataloader=valid_dataloader,
        run_evaluation_on_training_set=run_evaluation_on_training_set,
        iterative_mode=iterative_mode,
        precomputed_response_embeddings=precomputed_response_embeddings
    )

    # Save the trained model
    model.save_pretrained(path_to_output_dir)

    # Save the train configuration
    output_dir = Path(path_to_output_dir)
    config_file = output_dir / 'train_config.json'
    config_file.write_text(json.dumps(config, indent=2))

    # Save the train history
    history = pd.DataFrame({
        'epoch': range(1, config['epochs'] + 1),
        'train_loss': train_loss_history or None,
        'eval_loss': eval_loss_history or None
    })
    history.to_csv(str(output_dir / 'history.csv'), index=False)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
