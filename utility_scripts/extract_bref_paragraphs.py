from pathlib import Path
import re

import pandas as pd
import click
import docx

replacements = [
    # Remove "Figure X.Y" and similar stuff
    (r'(Table|Figure|Annex|Technique|activity)[\s\t]+[\s\dIVX]+[\.\d]*:?[\s\t]*', ''),
    # replace multiple consecutive tabs and/or spaces with a single space
    (r'[\s\t]+', ' '),
    # remove section headers
    (r'^[\s\t]*\d+\.(\d\.?)*', ''),
    # remove list items labels (e.g. a), b), c), ...) at the beginning
    # of the paragraph
    (r'^\(?([a-zA-Z\d•]{1}|[ivxIVX]+)[\)\.]?[\s\t]+', ''), 
    # remove references
    (r'^\[\d+\].*', '')
]

def clean_paragraph(paragraph_text):
    clean_text = paragraph_text.strip('.:,;-\t ')
    for (pattern, repl) in replacements:
        clean_text = re.sub(pattern, repl, clean_text)
    clean_text = clean_text.strip('.:,;-\t ')
    return clean_text

@click.command()
@click.argument(
    'input-files',
    type=click.Path(exists=True, dir_okay=False),
    nargs=-1,
    help='Path(s) to one or more BREF documents in .docx format.'
)
@click.argument(
    'output-path',
    type=click.Path(exists=False),
    help='Path where the extracted paragraphs will be saved.'
)
@click.option(
    '-l', '--min-paragraph-length',
    type=int,
    default=10,
    help='Minimum paragraph length (in words). Shorter ones will be discarded.'
)
def main(input_files, output_path, min_paragraph_length):
    all_paragraphs = []
    for input_file in input_files:
        # Load the document
        doc = docx.Document(input_file)
        # Clean the paragraphs
        cleaned_paragraphs = [clean_paragraph(p.text) for p in doc.paragraphs]
        # Filter out short paragraphs
        cleaned_paragraphs = list(
            filter(
                lambda p: len(p.split()) >= min_paragraph_length,
                cleaned_paragraphs
            )
        )
        filename = Path(input_file).stem
        all_paragraphs.extend((filename, p) for p in cleaned_paragraphs)
    
    # Create a DataFrame comprising all paragraphs and save it to disk
    df_all_paragraphs = pd.DataFrame(
        all_paragraphs, columns=['filename', 'paragraph']
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_all_paragraphs.to_csv(output_path, index=False)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
