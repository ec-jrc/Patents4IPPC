from functools import reduce
from pathlib import Path
import operator
import json
import re

from nltk.corpus import stopwords
import nltk
import click

ACRONYM_PATTERN = re.compile(r'[A-Z][A-Z]+')
REVERSE_ACRONYM_PATTERN = re.compile(r'[A-Z][A-Z]+\s+\([a-zA-Z-\s]+\)')

def capitalize_first_letter(s):
    return s[0].upper() + s[1:].lower()

def find_acronyms(doc, stopwords):
    """ Find acronyms of the form "Non-disclosure Agreement (NDA)"
    within the given string.

    Args:
        doc (str): The string in which to look for acronyms.
        stopwords (set): Set of stopwords that shouldn't be considered
          when forming an acronym. For instance, the acronym "BERT"
          means "Bidirectional Encoder Representations from
          Transformers", but the "F" doesn't appear in the acronym
          because "from" is a stopword.

    Returns:
        dict(str,str): Dictionary that maps the acronyms found in `doc`
          to their respective definitions.
    """

    acronyms = []
    tokens = nltk.word_tokenize(doc, language='english')
    for i, t in enumerate(tokens):
        false_acronym = False
        if ACRONYM_PATTERN.match(t) and (0 < i < len(tokens) - 1):
            if tokens[i-1] == '(' and tokens[i+1] == ')':
                j = i - 2
                k = len(t)
                words = []
                while j >= 0 and k > 0 and not false_acronym:
                    word = tokens[j]
                    subwords = word.split('-')[::-1]
                    words.append(word)
                    for subword in subwords:
                        if subword not in stopwords:
                            k -= 1
                            if (len(subword) == 0
                                    or subword[0].lower() != t[k].lower()):
                                false_acronym = True
                                break
                    j -= 1
                if k == 0 and not false_acronym:
                    acronym_elems = [
                        capitalize_first_letter(word)
                        if word not in stopwords
                        else word
                        for word in words[::-1]
                    ]
                    acronyms.append((t, ' '.join(acronym_elems)))
    return dict(acronyms)

def find_reverse_acronyms(doc, stopwords):
    """ Find acronyms of the form "NDA (Non-disclosure Agreement)"
    within the given string.

    Args:
        doc (str): The string in which to look for acronyms.
        stopwords (set): Set of stopwords that shouldn't be considered
          when forming an acronym. For instance, the acronym "BERT"
          means "Bidirectional Encoder Representations from
          Transformers", but the "F" doesn't appear in the acronym
          because "from" is a stopword.

    Returns:
        dict(str,str): Dictionary that maps the acronyms found in `doc`
          to their respective definitions.
    """

    acronyms = []
    for match in REVERSE_ACRONYM_PATTERN.finditer(doc):
        matched_str = doc[match.start():match.end()]
        # Find the acronym
        acronym_end_idx = re.search(r'\s', matched_str).start()
        acronym = matched_str[:acronym_end_idx]
        # Find the acronym definition (inside the parentheses)
        acronym_def_start_idx = matched_str.index('(') + 1
        acronym_def_end_idx = matched_str.index(')')
        acronym_def = matched_str[acronym_def_start_idx:acronym_def_end_idx]
        acronym_elems = acronym_def.split()
        acronym_subelems = list(map(lambda e: e.split('-'), acronym_elems))
        acronym_subelems = list(reduce(operator.add, acronym_subelems))
        # Check if the definition inside the parentheses is well-formed
        acronym_subelems_initials = [
            el[0].lower()
            if len(el) > 0
            else el
            for el in acronym_subelems
            if el not in stopwords
        ]
        acronym_letters = list(acronym.lower())
        if acronym_letters == acronym_subelems_initials:
            acronym_def_refined = ' '.join([
                capitalize_first_letter(el)
                if el not in stopwords
                else el
                for el in acronym_elems                
            ])
            acronyms.append((acronym, acronym_def_refined))
    return dict(acronyms)


@click.command()
@click.argument(
    'bref_docs',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    nargs=-1
)
@click.argument('output_dir', type=click.Path(), required=True, nargs=1)
def main(bref_docs, output_dir):
    en_stopwords = set(stopwords.words('english'))
    for doc in bref_docs:
        doc_content = Path(doc).read_text(encoding='utf-8')
        
        acronyms_map = {}
        acronyms_map.update(find_acronyms(doc_content, en_stopwords))
        acronyms_map.update(find_reverse_acronyms(doc_content, en_stopwords))

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get the name of the BREF document (assuming the filename starts
        # with an acronym denoting the document name)
        bref_name = re.findall(r'[A-Z][A-Z]+', Path(doc).stem)[0]
        
        output_path = output_dir / f'{bref_name}.json'
        output_path.write_text(
            json.dumps(acronyms_map, indent=2, sort_keys=True)
        )

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
