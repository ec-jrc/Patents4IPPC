from pathlib import Path

import pandas as pd
import joblib
import click

from patents4IPPC.evaluation import (
    compute_cosine_scores, compute_spearman_querywise, compute_ndcg
)
import utils


@click.command()
@click.option(
    '-mt', '--model-type',
    type=click.Choice(['tfidf', 'glove', 'use', 'huggingface', 'dual']),
    required=True,
    help='Type of model to use for indexing the responses.'
)
@click.option(
    '-mc', '--model-checkpoint', 'path_to_model_checkpoint',
    type=click.Path(exists=True),
    required=True,
    help='Path to a pre-trained model.'
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
    '-b', '--batch-size',
    type=int,
    default=2,
    help='Number of documents to encode at a time.'
)
@click.option(
    '-e', '--response-embeddings', 'path_to_response_embeddings',
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=('Path to a pickled dictionary containing pre-computed embeddings '
          'for the responses in the dataset. Required if --model-type="dual".')
)
def main(
    model_type,
    path_to_model_checkpoint,
    path_to_dataset,
    batch_size,
    path_to_response_embeddings
):
    if model_type == 'dual':
        assert path_to_response_embeddings is not None, \
               ('Must provide pre-computed response embeddings for evaluating '
                'the performances of a DualTransformer model.')
    
    embedder = utils.get_embedder(model_type, path_to_model_checkpoint)
    dataset = pd.read_csv(path_to_dataset).dropna(subset=['query', 'response'])

    precomputed_response_embeddings = None
    if path_to_response_embeddings is not None and model_type == 'dual':
        precomputed_response_embeddings = joblib.load(
            path_to_response_embeddings
        )

    df_cosine_scores = compute_cosine_scores(
        dataset, embedder, batch_size, precomputed_response_embeddings
    )

    spearman_score_mean, spearman_score_std = compute_spearman_querywise(
        df_cosine_scores
    )
    ndcg_score_mean, ndcg_score_std = compute_ndcg(df_cosine_scores)
    
    dataset_name = Path(path_to_dataset).stem
    print('Spearman rank on {}: {:.3f} (± {:.3f})\n'.format(
        dataset_name, spearman_score_mean, spearman_score_std
    ))
    print('NDCG score on {}: {:.3f} (± {:.3f})'.format(
        dataset_name, ndcg_score_mean, ndcg_score_std
    ))

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
