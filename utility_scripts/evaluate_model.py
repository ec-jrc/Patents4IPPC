from pathlib import Path

import pandas as pd
import joblib
import click

from patents4IPPC.evaluation import (
    compute_cosine_scores, compute_spearman_querywise, compute_ndcg
)
from patents4IPPC.embedders.utils import get_embedder


def convert_dataset_to_evaluation_format(path_to_dataset):
    dataset_dir = Path(path_to_dataset)
    qrels = pd.read_csv(
        str(dataset_dir / "qrels.txt"),
        header=None,
        names=["path_to_query", "path_to_response", "label"],
    )

    queries_dir = dataset_dir / "qs"
    queries_content = {
        query_file.stem: query_file.read_text().replace("\n", "[SEGMENT_SEP]")
        for query_file in queries_dir.iterdir()
    }

    responses_dir = dataset_dir / "rels"
    responses_content = {
        response_file.stem: response_file.read_text().replace("\n", "[SEGMENT_SEP]")
        for response_file in responses_dir.iterdir()
    }

    def extract_ids_and_contents(qrel):
        query_id = Path(qrel["path_to_query"]).stem
        query_content = queries_content[query_id]
        response_id = Path(qrel["path_to_response"]).stem
        response_content = responses_content[response_id]        
        return [query_id, query_content, response_id, response_content]
    qrels[["query_id", "query", "response_id", "response"]] = qrels.apply(
        extract_ids_and_contents, axis="columns", result_type="expand"
    )

    qrels = qrels.drop(columns=["path_to_query", "path_to_response"])
    return qrels


@click.command()
@click.option(
    '-mt', '--model-type',
    type=click.Choice(['tfidf', 'glove', 'use', 'huggingface', 'dual', 'hierarchical']),
    required=True,
    help='Type of model to use for indexing the responses.'
)
@click.option(
    '-mc', '--model-checkpoint', 'path_to_model_checkpoint',
    type=click.Path(),
    required=True,
    help='Path to a pre-trained model.'
)
@click.option(
    '-d', '--dataset', 'path_to_dataset',
    type=click.Path(exists=True),
    required=True,
    help=('Path to a dataset in .csv format with "standard" column names '
          '(i.e. it must have at least the following columns: query_id, '
          'query, response_id, response, label) OR path to a directory '
          'containing a dataset suitable for Hierarchical Transformer '
          '(i.e. it must contain a "qrels.txt" file and two subdirectories '
          'called "qs" and "rels").')
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
@click.option(
    '-o', '--output-path',
    type=click.Path(),
    default=None,
    help='Path to a file where predictions will be saved (in CSV format).'
)
def main(
    model_type,
    path_to_model_checkpoint,
    path_to_dataset,
    batch_size,
    path_to_response_embeddings,
    output_path
):
    if model_type == 'dual':
        assert path_to_response_embeddings is not None, \
               ('Must provide pre-computed response embeddings for evaluating '
                'the performances of a DualTransformer model.')
    
    embedder = get_embedder(model_type, path_to_model_checkpoint)
    if Path(path_to_dataset).is_dir():
        assert model_type == "hierarchical", \
            ("If you're not evaluating a Hierarchical Transformer model, you "
             "must provide a dataset in .csv format.")
        dataset = convert_dataset_to_evaluation_format(path_to_dataset)
    else:
        dataset = pd.read_csv(path_to_dataset)
    
    dataset = dataset.dropna(subset=['query', 'response'])

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

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_cosine_scores.to_csv(output_path, index=False)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
