import joblib
import click

import utils


@click.command()
@click.option(
    '-d', '--dataset', 'path_to_dataset',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help=('Path to a dataset with "standard" column names, i.e. it must have '
          'at least the following columns: query_id, query, response_id, '
          'response, label.')
)
@click.option(
    '-mt', '--model-type',
    type=click.Choice(['tfidf', 'glove', 'use', 'huggingface', 'dual']),
    required=True,
    help='Type of model to use for indexing the responses.'
)
@click.option(
    '-mc', '--model-checkpoint', 'path_to_model_checkpoint',
    type=click.Path(exists=True),
    default=None,
    help=('Path to a pre-trained model. Required unless "--model-type" is '
          '"tfidf", in which case there are two possibilities: either this '
          'parameter is provided, meaning that a pre-trained TF-IDF model is '
          'used to index the responses, or it is not, meaning that a fresh '
          'TF-IDF model is fitted on the responses, then it is used to index '
          'them.')
)
@click.option(
    '-b', '--batch-size',
    type=int,
    default=2,
    help='Number of responses to encode at a time.'
)
@click.option(
    '-o', '--output-path',
    type=click.Path(),
    required=True,
    help=('Location where the dictionary containing the response embeddings '
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
    path_to_dataset,
    model_type,
    path_to_model_checkpoint,
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

    # Load the dataset
    dataset, _, _ = utils.load_dataset(
        path_to_dataset, train_portion=1.0, valid_portion=0.0, test_portion=0.0
    )

    # Make sure that response IDs are integers (otherwise we can't save
    # them in a FAISS index)
    response_ids_dtype = dataset['response_id'].dtype
    assert response_ids_dtype == 'int64', \
           f'Response IDs must be integers (found "{response_ids_dtype}").'
    
    # Load the embedder
    embedder = utils.get_embedder(model_type, path_to_model_checkpoint)
    # Optionally fit the embedder
    dataset_unique_responses = dataset.drop_duplicates(
        subset=['response_id'], keep='first'
    )
    if model_type == 'tfidf' and path_to_model_checkpoint is None:
        embedder.fit(dataset_unique_responses['response'].values)
        joblib.dump(embedder, tfidf_output_path)

    # Embed the responses and create a dictionary
    utils.index_documents(
        documents=dataset_unique_responses['response'].values.tolist(),
        ids=dataset_unique_responses['response_id'].values,
        embedder=embedder,
        batch_size=batch_size,
        as_python_dictionary=True,
        store_on_disk=True,
        filename=output_path
    )

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
