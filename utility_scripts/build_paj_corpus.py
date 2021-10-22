from pathlib import Path

from bs4 import BeautifulSoup
import pandas as pd
import joblib
import click

@click.command()
@click.option(
    '-i', '--input-dir',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Path to a directory containing patent documents in .NRM format.'
)
@click.option(
    '-o', '--output-path',
    type=click.Path(exists=False),
    required=True,
    help='Path where the corpus will be saved.'
)
def main(input_dir, output_path):
    # Read the abstracts (Patent Abstracts of Japan, or PAJ)
    paj_dir = Path(input_dir)
    def extract_patent_info(filename):
        soup = BeautifulSoup(filename.read_text(), features='lxml')
        patent_num = soup.select_one('B110').get_text()      # INID code 11
        application_num = soup.select_one('B210').get_text() # INID code 21
        patent_title = soup.select_one('B542').get_text()    # INID code 54
        # Fix the title (often it is all uppercase)
        patent_title = patent_title[0].upper() + patent_title[1:].lower()
        patent_abstract = soup.select_one('SDOAB').get_text()
        return patent_num, application_num, patent_title, patent_abstract

    parallel_executor = joblib.Parallel(n_jobs=-1)
    pajs = parallel_executor(
        joblib.delayed(extract_patent_info)(filename)
        for filename in paj_dir.glob('**/*.NRM')
    )
    df_pajs = pd.DataFrame(
        pajs,
        columns=['patent_num', 'application_num', 'patent_title',
                 'patent_abstract']
    )
    # Save the extracted PAJs to disk
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_pajs.to_csv(output_path, index=False)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
