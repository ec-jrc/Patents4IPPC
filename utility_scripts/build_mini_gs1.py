from pathlib import Path
import pandas as pd
import click


@click.command()
@click.argument(
    'input-files',
    type=click.Path(exists=True, dir_okay=False),
    nargs=-1
)
@click.argument('output-path', type=click.Path(exists=False))
def main(input_files, output_path):
    qrels = []
    for input_file in input_files:
        filename = Path(input_file).stem
        df_responses = pd.read_excel(
            input_file,
            engine='openpyxl',
            usecols=[1, 2, 3],
            skiprows=[1]
        )
        qrels.extend(
            (filename, *info) for info in df_responses.itertuples(
                index=False, name=None
            )
        )
    df_qrels = pd.DataFrame(
        qrels,
        columns=[
            'bref_passage', 'abstract', 'relevance_score',
            'relevant_passages'
        ]
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_qrels.to_csv(output_path, index=False)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
