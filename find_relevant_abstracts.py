from pathlib import Path

import pandas as pd
import click

from patents4IPPC.similarity_search.faiss_ import FaissDocumentRetriever
import utils


@click.command()
@click.option(
    '-mt', '--model-type',
    type=click.Choice(['tfidf', 'glove', 'use', 'huggingface', 'dual']),
    required=True,
    help='Type of model to use for indexing the corpus.'
)
@click.option(
    '-mc', '--model-checkpoint', 'path_to_model_checkpoint',
    type=click.Path(exists=True),
    required=True,
    help='Path to a pre-trained model to use for finding relevant abstracts.'
)
@click.option(
    '-i', '--index', 'path_to_faiss_index',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help='Path to a FAISS index containing pre-computed response embeddings.'
)
@click.option(
    '-k', '--top-k', 'k',
    type=int,
    required=True,
    help='Number of relevant abstracts to retrieve for each query.'
)
@click.option(
    '-d', '--dataset', 'path_to_dataset',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help=('Path to a dataset that will be used to map the retrieved IDs to '
          'the corresponding abstracts.')
)
@click.option(
    '--id-column',
    required=True,
    help='Name of the dataset column that represents patent IDs.'
)
@click.option(
    '--abstract-column',
    required=True,
    help='Name of the dataset column that represents patent abstracts.'
)
@click.option(
    '-g', '--use-gpu',
    is_flag=True,
    help='Use a GPU for retrieving relevant abstracts with FAISS.'
)
@click.argument(
    'input_files',
    type=click.Path(exists=True, dir_okay=False),
    nargs=-1
)
@click.argument(
    'output_path',
    type=click.Path(exists=False)
)
def main(
    model_type,
    path_to_model_checkpoint,
    path_to_faiss_index,
    k,
    path_to_dataset,
    id_column,
    abstract_column,
    use_gpu,
    input_files,
    output_path
):
    # Read input files
    queries = [Path(f).read_text() for f in input_files]

    # Load an embedder
    print('Loading the embedder...')
    embedder = utils.get_embedder(model_type, path_to_model_checkpoint)

    # Embed the queries
    print('Embedding the queries...')
    query_embeddings = embedder.embed_documents(queries)
    del queries
    del embedder

    # Find the k closest abstracts for each query
    retriever = FaissDocumentRetriever(
        path_to_faiss_index, use_gpu=use_gpu, verbose=True
    )
    scores, ids = retriever.find_closest_matches(query_embeddings, k=k)
    del retriever

    # Use "ids" to retrieve the actual abstracts from the dataset
    print('Retrieving abstracts from IDs...')
    ids_flat = ids.reshape((-1,))
    dataset = pd.read_csv(
        path_to_dataset,
        index_col=id_column,
        encoding='latin1'
    )
    closest_abstracts = dataset.loc[ids_flat, abstract_column].values
    del dataset

    # Save the results to disk
    query_names = [Path(f).name for f in input_files for _ in range(k)]
    scores_flat = scores.reshape((-1,))
    results = pd.DataFrame({
        'query': query_names,
        'abstract': closest_abstracts,
        'score': scores_flat
    })
    results.to_csv(output_path, index=False)


if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
