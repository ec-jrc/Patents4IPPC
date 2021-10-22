from pathlib import Path
import json
import re

import pandas as pd
import click

from patents4IPPC import preprocessing

# replacements = [
#     # Remove "Figure X.Y" and similar stuff
#     (r'(Table|Figure|Annex|Technique|activity)[\s\t]+[\s\dIVX]+[\.\d]*:?[\s\t]*', ''),
#     # replace multiple consecutive tabs and/or spaces with a single space
#     (r'[\s\t]+', ' '),
#     # remove section headers
#     (r'^[\s\t]*\d+\.(\d\.?)*', ''),
#     # remove list items labels (e.g. a), b), c), ...) at the beginning
#     # of the paragraph
#     (r'^\(?([a-zA-Z\d•]{1}|[ivxIVX]+)[\)\.]?[\s\t]+', ''), 
#     # remove references
#     (r'^\[\d+\].*', '')
# ]

# def clean_paragraph(paragraph_text):
#     clean_text = paragraph_text.strip('.:,;-\t ')
#     for (pattern, repl) in replacements:
#         clean_text = re.sub(pattern, repl, clean_text)
#     clean_text = clean_text.strip('.:,;-\t ')
#     return clean_text

def clean_line(line, acronyms_map=None):
    # Remove references to figures etc
    line = re.sub(
        '^(Table|Figure|Annex|Technique|Source|activity).*',
        '',
        line
    )
    # Remove citations
    line = re.sub(r'\[.+?\]', '', line)
    # Remove numbered section headers
    line = re.sub(r'^\s*\d+\s*\.(\s*\d+\s*\.?)*.*', '', line)
    # Remove other known headers
    line = re.sub(
        r'^\s*([tT]echnical\s)?[dD]escription.*',
        '',
        line
    )
    # Normalize whitespace characters
    line = re.sub(r'\s', ' ', line)
    # Collapse multiple whitespaces
    line = re.sub(r'\s\s+', ' ', line)
    # Replace acronyms
    acronyms_map = acronyms_map or {}
    line = preprocessing.expand_acronyms(line, acronyms_map)

    return line.strip(r'.,:;-\t ')

@click.command()
@click.option(
    '-q', '--queries', 'path_to_queries_dir',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Path to a directory containing the queries.'
)
@click.option(
    '-a', '--acronyms-maps', 'path_to_acronyms_maps_dir',
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help='Path to a directory containing acronyms mappings in JSON format.'
)
@click.option(
    '-l', '--min-line-length',
    type=int,
    default=10,
    help='Minimum line length (in words). Shorter ones will be discarded.'
)
@click.option(
    '-o', '--output-path',
    type=click.Path(exists=False),
    required=True,
    help='Path where the corpus will be saved.'
)
def main(
    path_to_queries_dir,
    path_to_acronyms_maps_dir,
    min_line_length,
    output_path
):
    if path_to_acronyms_maps_dir is not None:
        acronyms_maps = {
            file_.stem: json.loads(file_.read_text())
            for file_ in Path(path_to_acronyms_maps_dir).glob('**/*.json')
        }
    else:
        acronyms_maps = {}
    
    all_lines = []
    for file_ in Path(path_to_queries_dir).glob('**/*.txt'):
        # Consider only numbered sections
        if not re.match(r'^\d.*', file_.name):
            continue

        lines = file_.read_text().split('\n')
        # Clean the lines
        bref_doc_id = file_.parent.name
        cleaned_lines = [
            clean_line(line, acronyms_maps.get(bref_doc_id, {}))
            for line in lines
        ]
        # Filter out short lines
        cleaned_lines = list(
            filter(lambda p: len(p.split()) >= min_line_length, cleaned_lines)
        )
        all_lines.extend((bref_doc_id, l) for l in cleaned_lines)
    
    # Create a DataFrame comprising all lines and save it to disk
    df_all_lines = pd.DataFrame(
        all_lines, columns=['bref_doc_id', 'text']
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_all_lines.to_csv(output_path, index=False)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
