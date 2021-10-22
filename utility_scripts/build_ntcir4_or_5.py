from pathlib import Path
import operator
import random
import re

from bs4 import BeautifulSoup
import pandas as pd
import click

from patents4IPPC import preprocessing


@click.command()
@click.option(
    '-c', '--corpus', 'path_to_corpus',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help='Path to the PAJ corpus.'
)
@click.option(
    '-q', '--queries', 'path_to_queries',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help='Path to a text file containing the queries (a.k.a. topics).'
)
@click.option(
    '-j', '--relevance-judgments', 'path_to_qrels',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help='Path to a text file containing the relevance judgments.'
)
@click.option(
    '-o', '--output-path',
    type=click.Path(exists=False),
    required=True,
    help='Path where the dataset will be saved.'
)
@click.option(
    '-p', '--preprocess', 'do_preprocess',
    is_flag=True,
    help='Do some preprocessing of the queries/responses.'
)
@click.option(
    '-t', '--use-titles',
    is_flag=True,
    help='Merge patent titles with their abstract.'
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
def main(
    path_to_corpus,
    path_to_queries,
    path_to_qrels,
    output_path,
    do_preprocess,
    use_titles,
    add_negative_examples,
    negatives_to_positives_ratio,
    seed
):
    # Fix the random seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    # Read the abstracts
    df_corpus = pd.read_csv(path_to_corpus, dtype=str)
    # Optionally merge titles and abstracts
    if use_titles:
        df_corpus.loc[:, 'patent_abstract'] = (
            df_corpus['patent_title'].apply(preprocessing.normalize_title)
            + df_corpus['patent_abstract']
        )
    df_corpus = df_corpus.drop(columns=['patent_title'])

    # Read the topics
    topics_file = Path(path_to_queries)
    soup = BeautifulSoup(topics_file.read_text(), features='lxml')
    claim_pattern = re.compile(r'\n?\(Claim \d+\)\s*')
    topics = []
    for topic_tag in soup.select('TOPIC'):
        # Extract topic number and text
        topic_num = topic_tag.select_one('NUM').get_text()
        topic_text = topic_tag.select_one('CLAIM').get_text()
        # Clean the topic text
        topic_text = claim_pattern.sub('', topic_text).strip()
        # Add the topic to the list of topics
        topics.append((topic_num, topic_text))
    df_topics = pd.DataFrame(topics, columns=['topic_num', 'topic_text'])
    
    # Read the qrels
    df_qrels = pd.read_csv(
        path_to_qrels,
        sep='\t',
        header=None,
        usecols=[0, 1, 2],
        names=['topic_num', 'relevance', 'patent_id'],
        dtype=str
    )
    
    # Convert the patent ID to a format that we can match with those
    # found in the corpus. In the case of NTCIR-4/5, we have to convert
    # it to a patent number (INID code 11)
    def convert_patent_id_to_patent_number(patent_id):
        # Patent IDs found in the qrels are like:
        #   PATENT-JA-UPA-1995-273578
        # We would like to convert them to:
        #   07273578
        # The last 6 digits are exactly the same as those found in the
        # qrels, whereas the first 2 correspond to the year expressed in
        # form of the Japanese era, specifically the Heisei era. Since
        # the Heisei era started in 1989, the year 1995 is written as 07
        # (seventh year of the Heisei era)
        year_str, code = patent_id.split('-')[-2:]
        year = int(year_str)
        if year >= 2000:
            # For some reason, years starting from 2000 are not encoded
            # as offsets from the beginning of the Heisei era
            return year_str + code
        offset_heisei = year - 1989 + 1
        return f'{offset_heisei:02d}{code}'
    
    df_qrels.loc[:, 'patent_num'] = df_qrels['patent_id'].apply(
        convert_patent_id_to_patent_number
    )
    
    # Extract the first letter of the relevance judgment (A, B, C or D).
    # We don't care about the rest
    df_qrels.loc[:, 'relevance'] = df_qrels['relevance'].apply(
        operator.itemgetter(0)
    )
    
    # Optionally add random negative examples
    if add_negative_examples:
        def find_negative_examples(patent_nums):
            positives = set(patent_nums)
            n_positives = len(positives)
            n_negatives = int(negatives_to_positives_ratio * n_positives)
            n_candidate_negatives = n_positives + n_negatives
            candidate_negatives_indices = random.sample(
                range(len(df_corpus)), k=n_candidate_negatives
            )
            candidate_negatives = set(
                df_corpus
                .iloc[candidate_negatives_indices]
                ['patent_num']
            )
            negatives = candidate_negatives.difference(positives)
            return list(negatives)[:n_negatives]
            
        df_positive_qrels = df_qrels.loc[df_qrels['relevance'].isin(['A', 'B'])]
        df_neg_examples = (
            df_positive_qrels
            .groupby('topic_num')['patent_num']
            .agg(find_negative_examples)
            .reset_index(drop=False)
            .rename(columns={'patent_num': 'negative_examples'})
        )
        additional_negative_qrels = [
            (topic_num, 'N', patent_num) # 'N' = synthetic neg. examples
            for topic_num, negative_examples in df_neg_examples.itertuples(
                index=False, name=None
            )
            for patent_num in negative_examples
        ]
        df_negative_qrels = pd.DataFrame(
            additional_negative_qrels,
            columns=['topic_num', 'relevance', 'patent_num']
        )
        # Add the synthetic negative qrels to the original qrels
        df_qrels = pd.concat([df_qrels, df_negative_qrels], ignore_index=True)

    # Convert string relevances to int
    labels_map = {'A': 2, 'B': 1, 'C': 0, 'D': 0, 'N': 0}
    df_qrels.loc[:, 'relevance'] = df_qrels['relevance'].map(labels_map)

    # Merge the three dataframes (patents, topics, qrels)
    df_topics_qrels = pd.merge(df_topics, df_qrels, on='topic_num')
    df_final = pd.merge(df_corpus, df_topics_qrels, on='patent_num')
    
    # Rename the columns
    df_final = df_final.rename(columns={
        'topic_num': 'query_id',
        'topic_text': 'query',
        'patent_num': 'response_id',
        'patent_abstract': 'response',
        'relevance': 'label'
    })

    # Pre-process queries and abstracts, if requested
    if do_preprocess:
        df_final.loc[:, 'query'] = df_final['query'].apply(
            preprocessing.clean_ntcir_3_to_5
        )
        df_final.loc[:, 'response'] = df_final['response'].apply(
            preprocessing.clean_ntcir_3_to_5
        )

    # Save the resulting DataFrame to disk
    relevant_columns = [
        'query_id', 'query', 'response_id', 'response', 'label'
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_final[relevant_columns].to_csv(output_path, index=False)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
