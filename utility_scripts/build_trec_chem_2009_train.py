import pandas as pd
import numpy as np
import click

@click.command()
@click.option(
    '-a', '--abstracts', 'path_to_abstracts',
    type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    '-q', '--prior-art-qrels', 'path_to_prior_art_qrels',
    type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    '-o', '--output-path',
    type=click.Path(exists=False)
)
def main(path_to_abstracts, path_to_prior_art_qrels, output_path):
    df_abstracts = pd.read_csv(path_to_abstracts)
    df_prior_art_qrels = pd.read_csv(path_to_prior_art_qrels)
    test_patent_ids = set(
        np.concatenate((
            df_prior_art_qrels['citing_patent_id'].values,
            df_prior_art_qrels['cited_patent_id'].values
        ))
    )
    df_abstracts_train = (
        df_abstracts
        .dropna(subset=['abstract'])
        .pipe(lambda df: df.loc[~df['patent_id'].isin(test_patent_ids)])
    )
    df_abstracts_train.to_csv(output_path, index=False)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
