import xml.etree.ElementTree as ET
from pathlib import Path
import operator
import re

from bs4 import BeautifulSoup
import pandas as pd
import joblib
import click

import utils


def convert_to_ucid(docnum):
    return re.sub(
        r'([A-Z]{2})(\d+)([A-Z]\d?)', lambda m: f'{m[1]}-{m[2]}-{m[3]}', docnum
    )

def extract_patent_info(path_to_patent_file, year):
    assert year in ['2009', '2010'], 'Specify either 2009 or 2010'
    try:
        root = ET.parse(path_to_patent_file).getroot()
    except ET.ParseError:
        if year == '2009':
            return (None,) * 4
        return (None,) * 6

    if year == '2010':
        family_id = root.attrib.get('family-id', '')

        citations = [
            cit.attrib['ucid']
            if 'ucid' in cit.attrib else convert_to_ucid(cit.attrib['dnum'])
            for cit in root.findall('.//patcit')
        ]
        citations_str = '|'.join(citations)

    title = root.find('.//invention-title[@lang="EN"]')
    if title is None:
        title = root.find('.//invention-title')
    title_lang = title.attrib.get('lang', '')
    
    abstract = root.find('.//abstract[@lang="EN"]')
    if abstract is None:
        abstract = root.find('.//abstract')
        if abstract is None:  # Sometimes the <abstract> tag is missing
            if year == '2009':
                return title.text, title_lang, None, None
            return family_id, title.text, title_lang, None, None, citations_str
    abstract_lang = abstract.attrib.get('lang', '')
    
    raw_abstract_tag = ET.tostring(abstract)
    abstract_text = (
        BeautifulSoup(raw_abstract_tag, features='lxml')
        .get_text()
        .strip()
        .replace('\n', '')
    )

    if year == '2009':
        return title.text, title_lang, abstract_text, abstract_lang

    return (family_id, title.text, title_lang, abstract_text, abstract_lang,
            citations_str)

@click.command()
@click.option(
    '-i', '--input-dir',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Path to a directory containing patent documents in XML format.'
)
@click.option(
    '-y', '--year',
    type=click.Choice(['2009', '2010']),
    required=True,
    help='Used to specify which corpus (2009 or 2010) to build.'
)
@click.option(
    '-o', '--output-path',
    type=click.Path(exists=False),
    required=True,
    help='Path where the corpus will be saved.'    
)
def main(input_dir, year, output_path):
    path_to_data = Path(input_dir)
    print('Acquiring the list of files...')
    all_patents = list(path_to_data.glob('**/*.xml'))
    print('List of files acquired.')

    parallel_executor = joblib.Parallel(n_jobs=-1)
    # Extract IDs for each patent document
    patent_ids = parallel_executor(
        joblib.delayed(operator.attrgetter('stem'))(p)
        for p in all_patents
    )
    # Extract the relevant information for each patent document
    patents_info = dict(zip(
        patent_ids,
        parallel_executor(
            joblib.delayed(extract_patent_info)(p, year)
            for p in all_patents
        )
    ))

    if year == '2009':
        column_names = [
            'patent_id',
            'title',
            'title_lang',
            'abstract',
            'abstract_lang'
        ]
    else:
        column_names = [
            'patent_id',
            'family_id',
            'title',
            'title_lang',
            'abstract',
            'abstract_lang',
            'citations'
        ]
    df_patents = pd.DataFrame(
        [(id_, *info)
          for id_, info in patents_info.items()],
        columns=column_names
    )

    # Provide an integer ID too for patents. It will be useful to easily
    # match patents with pre-computed embeddings stored in a FAISS index
    df_patents['patent_id_int'] = df_patents['patent_id'].apply(
        utils.ucid_to_int
    )

    # Save the final DataFrame to disk
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_patents.to_csv(output_path, index=False)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
