from pathlib import Path
import json

import pandas as pd
import click

from patents4IPPC import preprocessing


@click.command()
@click.option(
    '-c', '--corpus', 'path_to_corpus',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help='Path to the PatStat corpus.'
)
@click.option(
    '-b', '--bref-passages-dir', 'path_to_bref_passages_dir',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Path to a directory containing the BREF passages.'
)
@click.option(
    '-j', '--relevance-judgments', 'path_to_qrels',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help='Path to an XLSX file containing the relevance judgments.'
)
@click.option(
    '-a', '--acronyms-maps-dir', 'path_to_acronyms_maps_dir',
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help=('Path to a directory containing acronyms mappings in JSON format. '
          'If provided, it will be used to expand acronyms within BREF '
          'passages.')
)
@click.option(
    '-af', '--expand-acronyms-in-filenames',
    is_flag=True,
    help=('Expand acronyms in the filenames of the queries too. Ignored if '
          '"--acronyms-maps-dir" was not specified.')
)
@click.option(
    '-p', '--preprocess', 'do_preprocess',
    is_flag=True,
    help='Do some preprocessing of the abstracts.'
)
@click.option(
    '-t', '--use-titles',
    is_flag=True,
    help='Merge patent titles with their abstract.'
)
@click.option(
    '-s', '--segment-bref-passages',
    is_flag=True,
    help=('Separate BREF passages into individual segments, where one '
          'line corresponds to one segment. The segments will be '
          'separated from each other by a special sequence of '
          'characters ("[SEGMENT_SEP]").')
)
@click.option(
    '-o', '--output-path',
    type=click.Path(exists=False),
    required=True,
    help='Path where the dataset will be saved.'
)
def main(
    path_to_corpus,
    path_to_bref_passages_dir,
    path_to_qrels,
    path_to_acronyms_maps_dir,
    expand_acronyms_in_filenames,
    do_preprocess,
    use_titles,
    segment_bref_passages,
    output_path
):
    # Read the PatStat corpus
    df_corpus = pd.read_csv(
        path_to_corpus,
        usecols=['APPLN_ID', 'APPLN_TITLE', 'APPLN_ABSTR'],
        encoding='latin-1'
    )
    # Optionally merge titles and abstracts
    if use_titles:
        df_corpus.loc[:, 'APPLN_ABSTR'] = (
            df_corpus['APPLN_TITLE'].apply(preprocessing.normalize_title)
            + df_corpus['APPLN_ABSTR']
        )
    df_corpus = df_corpus.drop(columns=['APPLN_TITLE'])

    if path_to_acronyms_maps_dir is not None:
        # Read acronyms maps
        acronyms_maps = {
            map_file.stem: json.loads(map_file.read_text())
            for map_file in Path(path_to_acronyms_maps_dir).glob('**/*.json')
        }
    else:
        acronyms_maps = {}

    # Read and clean BREF passages
    bref_passages = {
        f'{file_.parent.name}:::{file_.name}':
            preprocessing.clean_bref_passage(
                bref_passage=file_.read_text(encoding='utf-8-sig'),
                acronyms_map=acronyms_maps.get(file_.parent.name, None),
                one_line_one_segment=segment_bref_passages
            )
        for file_ in Path(path_to_bref_passages_dir).glob('**/*.txt')
    }

    # Read the qrels
    df_qrels = pd.read_excel(
        path_to_qrels,
        usecols=['Query title', 'State', 'Text id', 'Score',
                 'Important part for the decision'],
        engine='openpyxl'
    )

    # Drop incomplete qrels
    df_qrels = df_qrels.dropna(subset=['Score'])

    # If several text fragments within a query are deemed relevant by
    # the experts, the dataset will contain a series of "almost"
    # duplicated rows where the only thing that varies is the value of
    # the "Important part for the decision" column (representing a
    # single relevant text fragment). Let's drop those duplicated rows
    # and join the relevant text fragments in a single string
    df_qrels = (
        df_qrels
        .groupby(['Query title', 'Text id']).agg({
            'State': lambda states: states.values[0],  # State is constant
            'Score': lambda scores: scores.values[0],  # Score is constant
            'Important part for the decision':
                lambda parts: '|'.join(
                    map(preprocessing.clean_patstat, parts.dropna().values)
                )
        })
        .reset_index(drop=False)
    )

    # Extract the content of the queries
    def transform_query_title(title):
        query_origin, query_fname = title.split(':::')
        if expand_acronyms_in_filenames:            
            query_fname = preprocessing.expand_acronyms(
                query_fname, acronyms_maps.get(query_origin, {})
            )
        return f'{query_origin}:::{query_fname}'      

    df_qrels['bref_passage'] = (
        df_qrels['Query title']
        .apply(transform_query_title)
        .map(bref_passages)
    )

    # Extract the content of the responses by merging df_qrels and
    # df_corpus
    df_gs1 = pd.merge(
        df_qrels, df_corpus, left_on='Text id', right_on='APPLN_ID'
    )

    # Clean the responses (i.e. patent abstracts), if requested
    if do_preprocess:
        df_gs1.loc[:, 'APPLN_ABSTR'] = df_gs1['APPLN_ABSTR'].apply(
            preprocessing.clean_patstat
        )
    
    # Convert query titles to integer IDs
    df_gs1['query_id'] = df_gs1['Query title'].astype('category').cat.codes

    # Rename and extract the relevant columns
    df_gs1 = (
        df_gs1
        .rename(columns={
            'Query title': 'query_title',
            'bref_passage': 'query',
            'APPLN_ID': 'response_id',
            'APPLN_ABSTR': 'response',
            'Score': 'label',
            'Important part for the decision': 'relevant_text_fragments'
        })
        .loc[:, ['query_title', 'query_id', 'query', 'response_id', 'response',
                 'label', 'relevant_text_fragments']]        
    )
    # Save GS1 to disk
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_gs1.to_csv(output_path, index=False)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
