from xml.etree import ElementTree as ET
from pathlib import Path

import pandas as pd
import click

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
    '-o', '--output-path',
    type=click.Path(exists=False),
    required=True,
    help='Path where the qrels will be saved.'
)
def main(path_to_queries, path_to_qrels, output_path):
    # Read topics
    root = ET.parse(path_to_queries)
    topics = root.findall('topic')
    topic_to_patents = {}
    for topic in topics:
        files = topic.findall('file')
        patent_ids = [
            Path(file_.text.strip()).stem
            for file_ in files
        ]
        topic_id = topic.findtext('number')
        topic_to_patents[topic_id] = patent_ids
    # Read "raw" qrels ("raw" refers to qrels of the form
    # <topic_id, patent_id>)
    df_qrels = pd.read_csv(
        path_to_qrels,
        header=None,
        sep=' ',
        names=['topic_id', 'other_patent_id'],
        usecols=[0, 2]
    )
    # Form "explicit" qrels ("explicit" refers to qrels of the form
    # <topic_patent_id, other_patent_id, relevance>. Note that
    # "relevance" will always be 1 since all we have are positive qrels)
    positive_pa_qrels = [
        (topic_patent_id, other_patent_id, 1)
        for topic_id, other_patent_id in df_qrels.itertuples(
            index=False, name=None
        )
        for topic_patent_id in topic_to_patents[topic_id]
    ]
    df_pa_qrels = pd.DataFrame(
        positive_pa_qrels,
        columns=['topic_patent_id', 'other_patent_id', 'relevance']
    )
    # Save the qrels to disk
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_pa_qrels.to_csv(output_path, index=False)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
