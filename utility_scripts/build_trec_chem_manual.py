from xml.etree import ElementTree
from pathlib import Path
import re

import pandas as pd
import click

from patents4IPPC import preprocessing
import utils


def clean_topic(topic_text):
    if topic_text is None:
        return None
    return re.sub(r'\s+', ' ', topic_text).replace('\n', '').strip()

@click.command()
@click.option(
    '-q', '--queries', 'path_to_queries',
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help='Path to an XML file containing the queries (a.k.a. topics).'
)
@click.option(
    '-j', '--relevance-judgments', 'path_to_qrels',
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help='Path to a text file containing the relevance judgments.'
)
@click.option(
    '-c', '--corpus', 'path_to_corpus',
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help=('Path to a TREC-Chem corpus (either 2009 or 2010, depending on '
          'which dataset you want to build).')
)
@click.option(
    '-y', '--year',
    type=click.Choice(['2009', '2010']),
    required=True,
    help='Used to specify which dataset (2009 or 2010) to build.'
)
@click.option(
    '-n', '--no-negative-scores',
    is_flag=True,
    help=('Discard relevance judgments for which the score is -1 (unjudged) '
          'or -2 (unsure).')
)
@click.option(
    '-p', '--preprocess', 'do_preprocess',
    is_flag=True,
    help='Do some preprocessing of the responses (i.e. patent abstracts).'
)
@click.option(
    '-t', '--use-titles',
    is_flag=True,
    help=('When forming queries and responses, merge their titles with their '
          'bodies.')
)
@click.option(
    '-o', '--output-path',
    type=click.Path(exists=False),
    required=True,
    help='Path where the dataset will be saved.'
)
def main(
    path_to_queries,
    path_to_qrels,
    path_to_corpus,
    year,
    no_negative_scores,
    do_preprocess,
    use_titles,
    output_path
):
    # Read the topics
    root = ElementTree.parse(path_to_queries)
    topics = root.findall('topic')
    df_topics = pd.DataFrame([
        (topic.findtext('number'),
         clean_topic(topic.findtext('title')),
         clean_topic(topic.findtext('narrative')))
        for topic in topics
    ], columns=['topic_id', 'topic_title', 'topic_narrative'])
    
    # Optionally merge topic titles and narratives
    if use_titles:
        df_topics.loc[:, 'topic_narrative'] = (
            df_topics['topic_title'].apply(preprocessing.normalize_title)
            + df_topics['topic_narrative']
        )
    df_topics = df_topics.drop(columns=['topic_title'])

    # Read the qrels
    df_qrels = pd.read_csv(
        path_to_qrels,
        sep=' ',
        names=['topic_id', 'document_id', 'relevance_score'],
        usecols=[0, 2, 4]
    )
    if no_negative_scores:
        # Filter out relevence judgments where the given score is
        # negative (-1 means "unjudged", -2 means "unsure")
        df_qrels = df_qrels[df_qrels['relevance_score'] >= 0]

    # Filter out non-patent documents
    patent_id_pattern = r'(EP|US|WO).*'
    df_qrels = (
        df_qrels
        .loc[df_qrels['document_id'].apply(
             lambda id_: re.match(patent_id_pattern, id_) is not None)]
        .rename(columns={'document_id': 'patent_id'})
        .reset_index(drop=True)
    )

    # Read the abstracts
    df_corpus = pd.read_csv(path_to_corpus).dropna(subset=['abstract'])

    # Optionally merge patent titles and abstracts
    if use_titles:
        df_corpus.loc[:, 'abstract'] = (
            df_corpus['title'].apply(preprocessing.normalize_title)
            + df_corpus['abstract']
        )
    df_corpus = df_corpus.drop(columns=['title']) 

    if year == '2009':  # If dealing with TREC-Chem 2009
        df_corpus['patent_number'] = df_corpus['patent_id'].apply(
            lambda pat_id: '-'.join(pat_id.split('-')[:2])
        )

        def pick_one_abstract(abstracts):
            first_valid_index = abstracts.first_valid_index()
            if first_valid_index is None:
                return float('nan')
            return abstracts[first_valid_index]

        grouping = df_corpus.groupby('patent_number')
        df_corpus = (
            grouping['abstract']  # Discard patent_id
            .agg(pick_one_abstract)
            .reset_index(drop=False)
            .rename(columns={'patent_number': 'patent_id'})
        )

    # Join topics, qrels and abstracts to form a single DataFrame
    df_topics_qrels = pd.merge(
        df_topics, df_qrels, left_on='topic_id', right_on='topic_id'
    )
    df_final = pd.merge(
        df_corpus,
        df_topics_qrels,
        left_on='patent_id',
        right_on='patent_id'
    )
    df_final = (
        df_final
        .rename(columns={'abstract': 'patent_abstract'})
        .loc[:, ('topic_id', 'topic_narrative', 'patent_id',
                 'patent_abstract', 'relevance_score')]
        .rename(columns={
            'topic_id': 'query_id',
            'topic_narrative': 'query',
            'patent_id': 'response_id',
            'patent_abstract': 'response',
            'relevance_score': 'label'
        })
    )

    # Pre-process the patent abstracts, if requested
    if do_preprocess:
        df_final.loc[:, 'query'] = df_final['query'].apply(
            preprocessing.clean_trec_chem
        )
        df_final.loc[:, 'response'] = df_final['response'].apply(
            preprocessing.clean_trec_chem
        )

    # Convert response IDs to integers so that they can be easily
    # matched with pre-compute embeddings stored in FAISS indexes
    df_final.loc[:, 'response_id'] = df_final['response_id'].apply(
        utils.ucid_to_int
    )

    # Save the resulting DataFrame to disk
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
