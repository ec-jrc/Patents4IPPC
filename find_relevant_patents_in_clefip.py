from pathlib import Path

import numpy as np
import pandas as pd
import click

from clef_ip_utils import extract_content_from_patent
from patents4IPPC.embedders.utils import get_embedder
from patents4IPPC.similarity_search.faiss_ import FaissDocumentRetriever


@click.command()
@click.option(
    '-mt', '--model-type',
    type=click.Choice([
        'tfidf', 'glove', 'use', 'huggingface', 'dual', 'hierarchical'
    ]),
    required=True,
    help=('Type of model to use for indexing the queries. Note that it should '
          'be the same model used to create the FAISS index.')
)
@click.option(
    '-mc', '--model-checkpoint', 'path_to_model_checkpoint',
    type=click.Path(exists=True),
    required=True,
    help=('Path to a pre-trained model to use for finding relevant patents. '
          'Note that it should be the same model used to create the FAISS '
          'index.')
)
@click.option(
    "-p", "--pooling-mode",
    type=click.Choice(["cls", "max", "mean"]),
    default=None,
    help=("Pooling strategy for aggregating token embeddings into sentence "
          "embeddings. Required only when \"--model-type\" is \"huggingface\" "
          "or \"dual\".")
)
@click.option(
    '-i', '--index', 'path_to_faiss_index',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help='Path to a FAISS index containing pre-computed embeddings of patents.'
)
@click.option(
    '-k', '--top-k', 'k',
    type=int,
    required=True,
    help='Number of relevant patents to retrieve for each query.'
)
@click.option(
    "-c", "--corpus-dir", "path_to_corpus_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help=("Directory containing a CLEF-IP corpus. Needed to extract the "
          "content of relevant patents.")
)
@click.option(
    '-f', '--files-list', 'path_to_files_list',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help=('Path to a text file containing the list of indexed patent files. '
          'Needed to map embedding IDs to actual file names. Must contain one '
          'file path per line, where said path must be relative to the corpus '
          'directory.')
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
    pooling_mode,
    path_to_faiss_index,
    k,
    path_to_corpus_dir,
    path_to_files_list,
    use_gpu,
    input_files,
    output_path
):
    # Read input files
    queries = [Path(f).read_text() for f in input_files]

    # Load an embedder
    print('Loading the embedder...')
    embedder = get_embedder(model_type, path_to_model_checkpoint, pooling_mode)

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

    # Use "ids" to retrieve the actual file names from the list
    with open(path_to_files_list, "r") as fp:
        files_list = np.array(fp.read().strip().split("\n"))
    print('Retrieving file names from IDs...')
    ids_flat = ids.reshape((-1,))
    closest_patents = files_list[ids_flat]

    # Extract the content of each relevant patent
    closest_patents_contents = []
    for patent in closest_patents:
        abstract, claims = extract_content_from_patent(patent)
        if model_type == "hierarchical":
            section_separator = "[SEGMENT_SEP]"
        else:
            section_separator = " "
        closest_patents_contents.append(
            section_separator.join([abstract] + claims)
        )

    # Save the results to disk
    query_names = [Path(f).name for f in input_files for _ in range(k)]
    scores_flat = scores.reshape((-1,))
    results = pd.DataFrame({
        'query': query_names,
        'patent_path': closest_patents,
        'patent_content': closest_patents_contents,
        'score': scores_flat
    })
    results.to_csv(output_path, index=False)


if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
