# The idea is to gather all the different preprocesing functions for the
# various datasets here

import operator
import re

from bs4 import BeautifulSoup
import pandas as pd

def normalize_title(title):
    if pd.isna(title):
        return ''
    return title[0].upper() + title[1:].lower() + '. '

def clean_abstract(abstract):
    # Remove junk that is commonly found in patent abstracts
    abstract = re.sub(r'PURPOSE:\s*', '', abstract)
    abstract = re.sub(r'PROBLEM TO BE SOLVED:\s*', '', abstract)
    abstract = re.sub(r'[\[\(]PROBLEMS?( TO BE SOLVED)?[\)\]]\s*', '', abstract)
    abstract = re.sub(r'SOLUTION:\s*', '', abstract)
    abstract = re.sub(r'CONSTITUTION:\s*', '', abstract)
    abstract = re.sub(r'TECHNICAL (FIELD|PROBLEM):\s*', '', abstract)
    abstract = re.sub(
        r'\[?MEANS (FOR|TO) (RE)?SOLV(ING|E) (THE )?PROBLEMS?\]?\s*',
        '',
        abstract
    )
    abstract = re.sub(
        r'SELECTED DRAWING:\s*(None|Figure\s[\.\w]+)', '', abstract
    )
    abstract = re.sub(r'COPYRIGHT:\s*.*', '', abstract)
    return abstract

def handle_newlines_and_whitespaces(text, one_line_one_segment=False):
    # Remove leading and trailing spaces/newlines/tabs
    text = re.sub(r'^\s*(.*?)\s*$', lambda match: match.groups()[0], text)
    # Remove newlines (as well as any preceding or following whitespaces)
    replacement_for_newlines = '[SEGMENT_SEP]' if one_line_one_segment else ' '
    text = re.sub(r'(\s*\n\s*)+', replacement_for_newlines, text)
    # Normalize whitespace characters
    text = re.sub(r'\s', ' ', text)
    # Collapse multiple whitespaces
    text = re.sub(r'\s\s+', ' ', text)
    return text.strip()

def expand_acronyms(text, acronyms_map):
    """
    We will replace any acronym found in `text` that's also included in
    `acronyms_map` as follows:
    - Acronyms of the form "Axxx Byyy Czzz (ABC)" (a.k.a. Normal
      acronyms) will be kept as they are, but the parentheses and their
      content will be removed
    - Acronyms of the form "ABC (Axxx Byyy Czzz)" (a.k.a. Reverse
      acronyms) will be replaced by the expanded definition of "ABC"
      (the parentheses and their content will be removed)
    - Free-roaming acronyms (like ABC) will be replaced by their
      expanded definition
    """

    for acronym, acronym_definition in acronyms_map.items():
        # Sometimes, acronym definitions appear in multiple variants.
        # For instance, the "-" (as in "Flue-gas Recirculation") may be
        # replaced by a whitespace (as in "Flue gas Recirculation")
        acronym_definition_alt = acronym_definition.replace('-', ' ')
        # Replace "Normal" acronyms with their definition
        text = re.sub(
            fr'({acronym_definition}|{acronym_definition_alt})\s+\({acronym}\)',
            acronym_definition,
            text,
            flags=re.IGNORECASE
        )
        # Replace "Reverse" acronyms with their definition
        text = re.sub(
            fr'{acronym}\s+\(({acronym_definition}|{acronym_definition_alt})\)',
            acronym_definition,
            text,
            flags=re.IGNORECASE
        )
        # Replace "Free-roaming" acronyms with their definition
        text = re.sub(
            fr'\b{acronym}\b',
            acronym_definition,
            text
        )
    return text

def clean_bref_passage(
    bref_passage, acronyms_map=None, one_line_one_segment=False
):
    # Remove section title
    bref_passage = re.sub(r'^(\d+\s*\.)+\d+\s+.+\n', '', bref_passage)
    # Remove description header
    bref_passage = re.sub(
        r'^\s*([tT]echnical\s)?[dD]escription\n',
        '',
        bref_passage,
        flags=re.MULTILINE
    )
    # Remove source reference
    bref_passage = re.sub(
        r'^\s*Source:.*\n', '', bref_passage, flags=re.MULTILINE
    )
    # Remove figure reference
    bref_passage = re.sub(
        r'^\s*Figure\s+(\d+\s*\.)+\d+\s*:.*\n',
        '',
        bref_passage,
        flags=re.MULTILINE
    )
    # Remove citations
    bref_passage = re.sub(r'\s\[\s?\d+,.+?\][,\.]?', '', bref_passage)
    # Remove bullet points
    bref_passage = re.sub('•', '', bref_passage)
    # Deal with newlines and whitespaces
    bref_passage = handle_newlines_and_whitespaces(
        bref_passage, one_line_one_segment
    )
    if acronyms_map is not None:
        # Expand acronyms
        bref_passage = expand_acronyms(bref_passage, acronyms_map)
    
    return bref_passage

def clean_helmers(text):
    # Apply standard abstract preprocessing
    text = clean_abstract(text)
    # TODO: Optionally add some Helmers-specific preprocessing
    # Deal with newlines and whitespaces
    text = handle_newlines_and_whitespaces(text)
    return text

def clean_patstat(text):
    # Apply standard abstract preprocessing
    text = clean_abstract(text)
    # TODO: Optionally add some PatStat-specific preprocessing
    # Deal with newlines and whitespaces
    text = handle_newlines_and_whitespaces(text)
    return text

def clean_ntcir_3_to_5(text):
    # Apply standard abstract preprocessing
    text = clean_abstract(text)
    # Remove mentions to chemical formulas
    text = re.sub(
        r'[\[\(]chemical formulas?( \d+( to \d+)?)?[\]\)]',
        '',
        text,
        flags=re.IGNORECASE
    )
    # Deal with newlines and whitespaces
    text = handle_newlines_and_whitespaces(text)
    return text

def clean_trec_chem(text):
    # Apply standard abstract preprocessing
    text = clean_abstract(text)    
    # Remove references to chemical formulas
    text = re.sub('##.+?##', '', text)
    # Normalize degrees symbol
    text = re.sub('<o>', '°', text)
    # Remove misspelled <i>...</i> tags
    text = re.sub(r'<i>(.+?)<\\i>', operator.itemgetter(1), text)
    # Remove sub/superscript junk (e.g. X.sub.Y or X.sup.Y)
    text = re.sub('.su[bp].', '', text, flags=re.IGNORECASE)
    # Normalize fractions (<fra>X<over>Y</fra> -> X/Y)
    text = re.sub(
        '<fra>(.+?)<over>(.+?)</fra>',
        lambda m: f'{m[1]}/{m[2]}',
        text,
        flags=re.IGNORECASE
    )
    # Remove leftover XML/HTML junk, including:
    # - Opened-and-never-closed tags like <IMAGE>, <MATH> and <CHEM>
    # - Non-matching opening and closing tags like <bold>...</highlight>
    #   (only the raw inner text is kept)
    # - Regular tags (only the raw inner text is kept)
    text = BeautifulSoup(text, features='lxml').get_text()
    # Deal with newlines and whitespaces
    text = handle_newlines_and_whitespaces(text)
    return text

def preprocess_for_tfidf(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)               # Non-alphanumeric
    text = re.sub(r'\b[a-zA-Z]{1}\b', ' ', text)  # Single letters
    text = re.sub(r'\b[0-9]+\b', ' ', text)       # Numbers
    text = re.sub(r'\s+', ' ', text)              # Multiple whitespaces
    text = text.strip()
    return text
