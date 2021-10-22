from pathlib import Path
import csv

import pandas as pd
import click


@click.command()
@click.option(
    '-d', '--dataset', 'path_to_dataset',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help='Path to a TSV file containing the STSb dataset.'
)
@click.option(
    '-o', '--output-path',
    type=click.Path(exists=False),
    required=True,
    help='Path where the dataset will be saved.'
)
def main(path_to_dataset, output_path):
    # Load the STSb dataset
    df_stsb = pd.read_csv(
        path_to_dataset, sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE
    )
    
    # Convert sentences to unicode. If you don't perform this step,
    # things like fitting a TfIdfVectorizer won't work
    df_stsb.loc[:, ['sentence1', 'sentence2']] = \
        df_stsb[['sentence1', 'sentence2']].astype('U').values
    
    # Rename columns
    df_stsb = df_stsb.rename(columns={
        'sentence1': 'query',
        'sentence2': 'response',
        'score': 'label'
    })
    
    # Give queries and responses unique integer IDs
    unique_texts = set(
        df_stsb['query'].dropna().values.tolist()
        + df_stsb['response'].dropna().values.tolist()
    )
    id_map = dict(zip(unique_texts, range(len(unique_texts))))
    df_stsb['query_id'] = df_stsb['query'].map(id_map)
    df_stsb['response_id'] = df_stsb['response'].map(id_map)

    # Save the dataset to disk, keeping relevant columns only
    relevant_columns = [
        'split', 'query_id', 'query', 'response_id', 'response', 'label'
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_stsb[relevant_columns].to_csv(output_path, index=False)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
