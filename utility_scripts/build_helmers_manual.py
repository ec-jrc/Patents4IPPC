from pathlib import Path
import re

import pandas as pd
import numpy as np
import click

from patents4IPPC import preprocessing


@click.command()
@click.option(
    '-d', '--dataset-dir', 'path_to_dataset_dir',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Path to a directory containing the Helmers manual dataset.'
)
@click.option(
    '-p', '--preprocess', 'do_preprocess',
    is_flag=True,
    help='Do some preprocessing of the patent abstracts.'
)
@click.option(
    '-t', '--use-titles',
    is_flag=True,
    help=('Merge patent titles with their abstracts. Be aware that some of '
          'the titles in the Helmers dataset are truncated (they end with '
          'three dots (...)).')
)
@click.option(
    '-o', '--output-path',
    type=click.Path(exists=False),
    required=True,
    help='Path where the dataset will be saved.'
)
def main(path_to_dataset_dir, do_preprocess, use_titles, output_path):
    dataset_dir = Path(path_to_dataset_dir)
    corpus_info_dir = dataset_dir / 'corpus_info'
    # Read patent pairs with human relevance judgments
    qrels_file = corpus_info_dir / 'human_label_pairs.npy'
    qrels = np.load(qrels_file, encoding='latin1', allow_pickle=True).item()
    
    # Read patent texts (title + abstract)
    patent_texts_file = corpus_info_dir / 'patcorpus_abstract.npy'
    patent_texts = np.load(
        patent_texts_file, encoding='latin1', allow_pickle=True
    ).item()
    
    # Optionally merge titles and abstracts
    def title_repl_func(matches):
        return matches[1] + '. ' if use_titles else ''
    patent_texts = {
        patent_id: re.sub(r'(.*?)\n', title_repl_func, patent_text, count=1)
        for patent_id, patent_text in patent_texts.items()
    }
    
    # Pre-process the abstracts, if requested
    if do_preprocess:
        patent_texts = {
            patent_id: preprocessing.clean_helmers(patent_text)
            for patent_id, patent_text in patent_texts.items()
        }
    
    # Build a dataframe with all the necessary information. Patent IDs
    # are converted to integers using the observation that they are of
    # the form "US123456789", so we just discard the first two
    # characters and convert the rest of the string to an integer
    df_helmers_manual = pd.DataFrame([
        (int(qid[2:]), patent_texts[qid], int(rid[2:]), patent_texts[rid],
         relevance_score)
        for (qid, rid), relevance_score in qrels.items()
    ], columns=['query_id', 'query', 'response_id' ,'response', 'label'])
    
    # Save the dataframe to disk
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_helmers_manual.to_csv(output_path, index=False)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
