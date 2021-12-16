import pandas as pd
import joblib
import click

from patents4IPPC import preprocessing
from patents4IPPC.similarity_search.faiss_ import index_documents_using_faiss
import utils


@click.command()
@click.option(
    '-i', '--input-file', 'path_to_patstat_corpus',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help='Path to the PatStat corpus.'
)
@click.option(
    '-mt', '--model-type',
    type=click.Choice(['tfidf', 'glove', 'use', 'huggingface', 'dual']),
    required=True,
    help='Type of model to use for indexing the corpus.'
)
@click.option(
    '-mc', '--model-checkpoint', 'path_to_model_checkpoint',
    type=click.Path(exists=True),
    default=None,
    help=('Path to a pre-trained model. Required unless "--model-type" is '
          '"tfidf", in which case there are two possibilities: either this '
          'parameter is provided, meaning that a pre-trained TF-IDF model is '
          'used to index the corpus, or it is not, meaning that a fresh '
          'TF-IDF model is fitted on the corpus, then it is used to index it.')
)
@click.option(
    '-t', '--use-titles',
    is_flag=True,
    help=('Whether or not to consider titles in addition to abstracts when '
          'embedding a patent. If set, titles and abstracts will be merged '
          'together.')
)
@click.option(
    '-b', '--batch-size',
    type=int,
    default=2,
    help='Number of documents to encode at a time.'
)
@click.option(
    '-o', '--output-path',
    type=click.Path(),
    required=True,
    help=('Location where the FAISS index representing the PatStat corpus '
          'will be saved.')
)
@click.option(
    '--tfidf-output-path',
    type=click.Path(),
    default=None,
    help=('Location where the fitted TF-IDF model will be saved. Required '
          'when "--model-type" is "tfidf" and "--model-checkpoint" was not '
          'specified.')
)
def main(
    path_to_patstat_corpus,
    model_type,
    path_to_model_checkpoint,
    use_titles,
    batch_size,
    output_path,
    tfidf_output_path
):
    if model_type != 'tfidf':
        assert path_to_model_checkpoint is not None, \
               'Please provide a model checkpoint.'
    if model_type == 'tfidf' and path_to_model_checkpoint is None:
        assert tfidf_output_path is not None, \
               ('Please provide a path where the fitted TF-IDF model will be '
                'saved.')

    embedder = utils.get_embedder(model_type, path_to_model_checkpoint)

    patstat_corpus = pd.read_csv(path_to_patstat_corpus, encoding='latin1')
    patstat_corpus = patstat_corpus.dropna()

    # Optionally merge patent titles and abstracts
    if use_titles:
        patstat_corpus.loc[:, 'APPLN_ABSTR'] = (
            patstat_corpus['APPLN_TITLE'].apply(preprocessing.normalize_title)
            + patstat_corpus['APPLN_ABSTR']
        )

    # Preprocess abstracts
    patstat_corpus.loc[:, 'APPLN_ABSTR'] = \
        patstat_corpus['APPLN_ABSTR'].apply(preprocessing.clean_patstat)

    # Optionally fit the embedder
    if model_type == 'tfidf' and path_to_model_checkpoint is None:
        embedder.fit(patstat_corpus['APPLN_ABSTR'].values)
        joblib.dump(embedder, tfidf_output_path)

    # Embed the abstracts and create a FAISS index
    documents = patstat_corpus['APPLN_ABSTR'].values.tolist()
    ids = patstat_corpus['APPLN_ID'].values
    del patstat_corpus
    index_documents_using_faiss(
        documents=documents,
        ids=ids,
        embedder=embedder,
        batch_size=batch_size,
        store_on_disk=True,
        filename=output_path
    )

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
