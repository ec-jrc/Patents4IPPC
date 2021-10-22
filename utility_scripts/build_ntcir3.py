from pathlib import Path
import operator
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
    '-q', '--queries-dir', 'path_to_queries_dir',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Path to a directory containing the queries (a.k.a. topics).'
)
@click.option(
    '-j', '--relevance-judgments', 'path_to_qrels',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help='Path to a text file containing the relevance judgments.'
)
@click.option(
    '-o', '--output-path',
    type=click.Path(),
    required=True,
    help='Path where the dataset will be saved.'
)
@click.option(
    '-p', '--preprocess', 'do_preprocess',
    is_flag=True,
    help='Do some preprocessing of the responses (i.e. patent abstracts).'
)
@click.option(
    '-t', '--use-titles',
    is_flag=True,
    help='Merge patent titles with their abstract.'
)
@click.option(
    '-qp', '--query-parts',
    type=click.Choice(['headline', 'text', 'description', 'narrative']),
    multiple=True,
    required=True,
    help='Parts of the query (i.e. newspaper article) to use.'
)
def main(
    path_to_corpus,
    path_to_queries_dir,
    path_to_qrels,
    output_path,
    do_preprocess,
    use_titles,
    query_parts
):  
    # Read the abstracts
    df_corpus = pd.read_csv(path_to_corpus, dtype=str)
    # Optionally merge patent titles and abstracts
    if use_titles:
        df_corpus.loc[:, 'patent_abstract'] = (
            df_corpus['patent_title'].apply(preprocessing.normalize_title)
            + df_corpus['patent_abstract']
        )
    df_corpus = df_corpus.drop(columns=['patent_title'])

    # Read the topics
    topics_dir = Path(path_to_queries_dir)
    topics = []
    def clean_tag_text(text):
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    for topic_file in topics_dir.glob('**/*.sgml'):
        topic_num = topic_file.stem
        soup = BeautifulSoup(
            topic_file.read_text(encoding='utf-8', errors='replace'),
            features='lxml'
        )
        topic_tag = soup.select_one('TOPIC')
        # Extract topic info
        topic_headline = clean_tag_text(
            topic_tag.select_one('A-HEADLINE').get_text()
        )
        topic_text = clean_tag_text(
            topic_tag.select_one('A-TEXT').get_text()
        )
        topic_description = clean_tag_text(
            topic_tag.select_one('DESCRIPTION').get_text()
        )
        topic_narrative = clean_tag_text(
            topic_tag.select_one('NARRATIVE').get_text()
        )
        # Add the topic to the list of topics
        topics.append((
            topic_num,
            topic_headline,
            topic_text,
            topic_description,
            topic_narrative 
        ))

    # Form the topics dataframe
    df_topics = pd.DataFrame(
        topics,
        columns=['topic_num', 'topic_headline', 'topic_text',
                 'topic_description', 'topic_narrative']
    )

    # Build the query text by merging the parts requested by the user
    def merge_texts(*texts):
        return '. '.join([text.strip('.') for text in texts]) + '.'
    chosen_columns = [f'topic_{part}' for part in query_parts]
    df_topics['query'] = df_topics[chosen_columns].apply(
        lambda column_values: merge_texts(*column_values), axis=1
    )
    
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
    # found in the corpus. In the case of NTCIR-3, we have to convert it
    # to an application number (INID code 21)
    def convert_patent_id_to_application_number(patent_id):
        # Patent IDs found in the qrels are like:
        #   PATENT-KKH-G-H08-169636
        # We would like to convert them to:
        #   08169636
        # Note that, in the example above, H08 refers to the eigth year
        # of the Heisei era (which would be 1996)
        offset_heisei, code = patent_id.split('-')[-2:]
        return offset_heisei[1:] + code        
    df_qrels.loc[:, 'application_num'] = df_qrels['patent_id'].apply(
        convert_patent_id_to_application_number
    )
    
    # Extract the first letter of the relevance judgment (A, B, C or D).
    # We don't care about the rest
    df_qrels.loc[:, 'relevance'] = df_qrels['relevance'].apply(
        operator.itemgetter(0)
    )
    
    # Convert string relevances to int
    labels_map = {'A': 2, 'B': 1, 'C': 0, 'D': 0}
    df_qrels.loc[:, 'relevance'] = df_qrels['relevance'].map(labels_map)

    # Merge the three dataframes (patents, topics, qrels)
    df_topics_qrels = pd.merge(df_topics, df_qrels, on='topic_num')
    df_final = pd.merge(df_corpus, df_topics_qrels, on='application_num')
    # NOTE: We merge "df_corpus" and "df_topics_qrels" on
    # "application_num" because that's what patents are identified with
    # in the qrels. However, we choose "patent_num" as the response ID
    # (see instruction below) because "application_num" is not unique in
    # the PAJ corpus, whereas "patent_num" is.

    # Rename the columns
    df_final = df_final.rename(columns={
        'topic_num': 'query_id',
        'patent_num': 'response_id',
        'patent_abstract': 'response',
        'relevance': 'label'
    })

    # Pre-process abstracts, if requested
    if do_preprocess:
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
