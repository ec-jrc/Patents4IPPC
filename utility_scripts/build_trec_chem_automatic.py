from pathlib import Path
import random

import pandas as pd
import click

from patents4IPPC import preprocessing
import utils


@click.command()
@click.option(
    '-c', '--corpus', 'path_to_corpus',
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help=('Path to a TREC-Chem corpus (either 2009 or 2010, depending on '
          'which dataset you want to build).')
)
@click.option(
    '-j', '--relevance-judgments', 'path_to_qrels',
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help='Path to a text file containing the relevance judgments.'
)
@click.option(
    '-p', '--preprocess', 'do_preprocess',
    is_flag=True,
    help='Do some preprocessing of the patent abstracts.'
)
@click.option(
    '-t', '--use-titles',
    is_flag=True,
    help='Merge patent titles with their bodies.'
)
@click.option(
    '-n', '--add-negative-examples',
    is_flag=True,
    help='Add synthetic negative examples to the dataset.'
)
@click.option(
    '-r', '--negatives-to-positives-ratio',
    type=float,
    default=1.0,
    help=('Number of synthetic negative examples to add for each positive '
          'example. Ignored unless "--add-negative-examples" was specified. '
          'Defaults to 1.0 (i.e. creates a balanced dataset).')
)
@click.option(
    '-s', '--seed',
    type=int,
    default=None,
    help=('Random seed used to generate random negative examples. Ignored '
          'unless "--add-negative-examples" was specified.')
)
@click.option(
    '-o', '--output-path',
    type=click.Path(exists=False),
    required=True,
    help='Path where the dataset will be saved.'
)
def main(
    path_to_corpus,
    path_to_qrels,
    do_preprocess,
    use_titles,
    add_negative_examples,
    negatives_to_positives_ratio,
    seed,
    output_path
):
    # Fix the random seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    # Read the corpus
    df_corpus = pd.read_csv(
        path_to_corpus, usecols=['patent_id', 'title', 'abstract']
    ).dropna(subset=['abstract'])
    
    # Optionally merge titles and abstracts
    if use_titles:
        df_corpus.loc[:, 'abstract'] = (
            df_corpus['title'].apply(preprocessing.normalize_title)
            + df_corpus['abstract']
        )
    df_corpus = df_corpus.drop(columns=['title'])    

    # Read the relevance judgments
    df_qrels = pd.read_csv(path_to_qrels)

    # Optionally add random negative examples
    if add_negative_examples:
        def find_negative_examples(patent_ids):
            positives = set(patent_ids)
            n_positives = len(positives)
            n_negatives = int(negatives_to_positives_ratio * n_positives)
            n_candidate_negatives = n_positives + n_negatives
            candidate_negatives_indices = random.sample(
                range(len(df_corpus)), k=n_candidate_negatives
            )
            candidate_negatives = set(
                df_corpus
                .iloc[candidate_negatives_indices]
                ['patent_id']
            )
            negatives = candidate_negatives.difference(positives)
            return list(negatives)[:n_negatives]
            
        df_neg_examples = (
            df_qrels
            .groupby('topic_patent_id')['other_patent_id']
            .agg(find_negative_examples)
            .reset_index(drop=False)
            .rename(columns={'other_patent_id': 'negative_examples'})
        )
        additional_negative_qrels = [
            (topic_patent_id, patent_id, 0) # 0 = negative example
            for topic_patent_id, negative_examples in df_neg_examples.itertuples(
                index=False, name=None
            )
            for patent_id in negative_examples
        ]
        df_negative_qrels = pd.DataFrame(
            additional_negative_qrels,
            columns=['topic_patent_id', 'other_patent_id', 'relevance']
        )
        # Add the synthetic negative qrels to the original qrels
        df_qrels = pd.concat([df_qrels, df_negative_qrels], ignore_index=True)

    # Merge the two dataframes (abstracts, qrels)
    df_tmp = pd.merge(
        df_corpus, df_qrels, left_on='patent_id', right_on='topic_patent_id'
    )
    df_tmp = (
        df_tmp
        .drop(columns=['patent_id'])
        .rename(columns={'abstract': 'topic_patent_abstract'})
    )
    df_trec_chem_automatic = pd.merge(
        df_tmp, df_corpus, left_on='other_patent_id', right_on='patent_id'
    )
    df_trec_chem_automatic = (
        df_trec_chem_automatic
        .drop(columns=['patent_id'])
        .rename(columns={'abstract': 'other_patent_abstract'})
        .loc[:, ['topic_patent_id', 'topic_patent_abstract', 'other_patent_id',
                 'other_patent_abstract', 'relevance']]
        .rename(columns={
            'topic_patent_id': 'query_id',
            'topic_patent_abstract': 'query',
            'other_patent_id': 'response_id',
            'other_patent_abstract': 'response',
            'relevance': 'label'
        })
    )
    
    # Pre-process the abstracts, if requested
    if do_preprocess:
        df_trec_chem_automatic.loc[:, 'query'] = \
            df_trec_chem_automatic['query'].apply(
                preprocessing.clean_trec_chem
            )
        df_trec_chem_automatic.loc[:, 'response'] = \
            df_trec_chem_automatic['response'].apply(
                preprocessing.clean_trec_chem
            )

    # Convert patent IDs to integers so that they can be easily matched
    # with pre-compute embeddings stored in FAISS indexes
    df_trec_chem_automatic.loc[:, 'query_id'] = \
        df_trec_chem_automatic['query_id'].apply(utils.ucid_to_int)
    df_trec_chem_automatic.loc[:, 'response_id'] = \
        df_trec_chem_automatic['response_id'].apply(utils.ucid_to_int)        

    # Save the final dataframe to disk
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_trec_chem_automatic.to_csv(output_path, index=False)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
