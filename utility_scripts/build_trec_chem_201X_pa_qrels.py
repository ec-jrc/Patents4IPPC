import xml.etree.ElementTree as ET
from functools import reduce
from pathlib import Path
import operator

import pandas as pd
import click


@click.command()
@click.option(
    '-c', '--corpus', 'path_to_corpus',
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help=('Path to the TREC-Chem 2010 corpus (both TREC-Chem 2010 and 2011 '
          'datasets refer to the same patent corpus).')
)
@click.option(
    '-q', '--queries-dir', 'path_to_queries_dir',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Path to a directory containing the queries (a.k.a. topics).'
)
@click.option(
    '-e', '--english-only',
    is_flag=True,
    help=('Keep only those qrels where both the query and the response patent '
          'are in English (there exist some French and German patents).')
)
@click.option(
    '-o', '--output-path',
    type=click.Path(exists=False),
    required=True,
    help='Path where the qrels will be saved.'
)
def main(path_to_corpus, path_to_queries_dir, output_path, english_only):
    df_corpus = pd.read_csv(path_to_corpus)
    if english_only:
        df_corpus = df_corpus[df_corpus['abstract_lang'] == 'EN']
    # Group patents by family ID
    def get_unique_citations(citations):
        citations_list = citations.apply(
            lambda cs: cs.split('|') if not pd.isna(cs) else []
        )
        citations_list_flat = list(reduce(operator.add, citations_list))
        unique_citations = set(citations_list_flat)
        return list(unique_citations)
    df_families = (
        df_corpus
        .groupby('family_id')
        [['patent_id', 'citations']]
        .agg({
            'patent_id': lambda s: s.values.tolist(),
            'citations': get_unique_citations
        })
    )
    # Get topic patents and their family IDs
    def find_family_id(path_to_patent_document):
        root = ET.parse(path_to_patent_document).getroot()
        family_id = int(root.attrib.get('family-id', -1))
        return family_id
    topics_dir = Path(path_to_queries_dir)
    topic_patents = [  # List of <Patent UCID, Family ID> pairs
        (filename.stem, find_family_id(filename))
        for filename in topics_dir.glob('**/*.xml')
    ]
    # Follow the procedure described in the TREC-Chem 2010 paper to
    # build qrels for the Prior Art (PA) task
    # (https://trec.nist.gov/pubs/trec19/papers/CHEM.OVERVIEW.pdf, Section 3.1)
    # Since the TREC-Chem 2011 paper doesn't explicitly mention how
    # qrels for the PA task are constructed, we assume that the
    # procedure is the same as TREC-Chem 2010
    patent_ids = set(df_corpus['patent_id'].values)
    positive_pa_qrels = []
    for topic_patent_ucid, topic_patent_family_id in topic_patents:        
        f1_set = df_families.loc[topic_patent_family_id, 'citations']
        # Filter out citations that are not in the collection
        f1_set = list(filter(
            lambda patent_id: patent_id in patent_ids, f1_set
        ))
        # Find the family members of the patent documents in the F1 set
        relevant_patents_ids = []
        for patent_id in f1_set:
            family_id = (
                df_corpus
                .loc[df_corpus['patent_id'] == patent_id, 'family_id']
                .values[0]
            )
            family_members = df_families.loc[family_id, 'patent_id']
            relevant_patents_ids.extend(family_members)    
        # Get the family members of the topic patent
        topic_patent_family_members = (
            df_families
            .loc[topic_patent_family_id, 'patent_id']
        )
        # Build the final set of relevant patents
        f3_set = list(set(
            topic_patent_family_members + relevant_patents_ids
        ))
        # Remove the topic patent from the F3 set. Note that:
        # 1) The topic patent is, by definition, a member of its own
        #    family, so it will be contained in
        #    'topic_patent_family_members', which is a subset of F3,
        #    hence it will be contained in F3, too (actually, if the
        #    topic patent has a non-English abstract and this script was
        #    called with the "-e" flag, then it will be removed from the
        #    collection, meaning that it won't end up in F3).
        # 2) The topic patent *could* be cited by its own family
        #    members, in which case it would end up in F3
        # In TREC-Chem 2009 PA Qrels, it *sometimes* happens that a
        # topic patent is listed among the relevant patents for that
        # same topic patent, meaning that occasionally there are qrels
        # of the form <X,X>. Nonetheless, I think it makes a lot of
        # sense to discard such qrels, hence the instruction below:
        if topic_patent_ucid in f3_set:
            f3_set.remove(topic_patent_ucid)
        # Extend the QRels. Note that all we have are positive qrels, so
        # the relevance score (third element of the triplets below) is
        # fixed to 1
        positive_pa_qrels.extend(
            (topic_patent_ucid, relevant_patent_ucid, 1)
            for relevant_patent_ucid in f3_set
        )

    # Make a pandas DataFrame out of the qrels and save it to disk
    df_pa_qrels = pd.DataFrame(
        positive_pa_qrels,
        columns=['topic_patent_id', 'other_patent_id', 'relevance']
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_pa_qrels.to_csv(output_path, index=False)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
