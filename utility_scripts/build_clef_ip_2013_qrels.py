from functools import reduce
from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET

import click
import pandas as pd


def is_english_patent(path_to_patent):
    try:
        patent_document_tag = ET.parse(str(path_to_patent)).getroot()
        lang = patent_document_tag.attrib.get("lang", None)
        return lang == "EN"
    except (FileNotFoundError, ET.ParseError):
        return False

def has_abstract_and_claims(path_to_patent):
    try:
        patent_document_tag = ET.parse(str(path_to_patent)).getroot()
        has_abstract = patent_document_tag.find("abstract") is not None
        has_claims = patent_document_tag.find("claims") is not None
        return has_abstract and has_claims
    except ET.ParseError:
        return False

def get_relative_file_path_from_ucid(ucid):
    if ucid.startswith("EP"):
        return Path(
            "EP",
            f"00000{ucid[3]}",
            ucid[4:6],
            ucid[6:8],
            ucid[8:10],
            f"{ucid}.xml"
        )
    if ucid.startswith("WO"):
        return Path(
            "WO",
            f"00{ucid[3:7]}",
            ucid[7:9],
            ucid[9:11],
            ucid[11:13],
            f"{ucid}.xml"
        )
    raise ValueError(f"Invalid UCID: {ucid}.")

def filter_out_non_english_patents(dataset, tfiles_dir, corpus_dir):
    topic_patents = dataset["patent_ucid"]
    rel_patents = dataset["rel_patent_ucid"]

    topic_patents_en_mask = topic_patents.apply(
        lambda ucid: is_english_patent(tfiles_dir / f"{ucid}.xml")
    )
    rel_patents_en_mask = rel_patents.apply(
        lambda ucid: is_english_patent(
            corpus_dir / get_relative_file_path_from_ucid(ucid)
        )        
    )
    
    return dataset[topic_patents_en_mask & rel_patents_en_mask]

def find_random_patent_of_type_a(dir_):
    dir_entries = list(dir_.iterdir())
    if dir_entries[0].is_file():
        a1_or_a2_patents = (list(dir_.glob("*-A1.xml"))
                            + list(dir_.glob("*-A2.xml")))
        return a1_or_a2_patents[0].stem
    
    random_index = random.randrange(0, len(dir_entries))
    return find_random_patent_of_type_a(dir_entries[random_index])

def add_synthetic_negatives(dataset, corpus_dir, num_negatives_per_positive=2):
    num_negatives_to_find = len(dataset) * num_negatives_per_positive
    relevant_patents = dataset["rel_patent_ucid"]
    non_relevant_patents = []
    while num_negatives_to_find > 0:
        candidate_patent = find_random_patent_of_type_a(corpus_dir)
        if candidate_patent in relevant_patents:
            continue
        path_to_candidate_patent = \
            corpus_dir / get_relative_file_path_from_ucid(candidate_patent)
        if not (is_english_patent(path_to_candidate_patent)
                and has_abstract_and_claims(path_to_candidate_patent)):
            continue
        non_relevant_patents.append(candidate_patent)
        num_negatives_to_find -= 1

    topic_patents = dataset["patent_ucid"]
    negative_qrels = pd.DataFrame({
        "patent_ucid": pd.concat([topic_patents] * num_negatives_per_positive),
        "rel_patent_ucid": non_relevant_patents,
        "label": -1.0
    })
    positive_qrels = dataset.copy()
    positive_qrels["label"] = 1.0
    
    new_dataset = pd.concat([positive_qrels, negative_qrels], ignore_index=True)
    return new_dataset

def get_qrels(
    corpus_dir,
    topics_tfiles_dir,
    topics_file,
    qrels_file,
    num_negatives_per_positive=2
):
    # Read training topics into a DataFrame
    N_LINES_PER_TOPIC = 5
    topics_lines = topics_file.read_text().splitlines()
    topics_grouped_lines = [
        topics_lines[i:(i + N_LINES_PER_TOPIC)]
        for i in range(0, len(topics_lines), N_LINES_PER_TOPIC)
    ]
    topics = [
        ET.fromstringlist(["<root>", *single_topic_lines, "</root>"])
        for single_topic_lines in topics_grouped_lines
    ]
    topics_tids_and_patent_ids = [
        (t.findtext("tid"), t.findtext("tfile").replace(".xml", ""))
        for t in topics
    ]
    df_topics = pd.DataFrame(
        topics_tids_and_patent_ids, columns=["topic_id", "patent_ucid"]
    )

    # Read the qrels into a DataFrame
    df_qrels = pd.read_csv(
        str(qrels_file),
        sep=" ",
        header=None,
        usecols=[0, 1],
        names=["topic_id", "rel_patent_ucid"]
    )
    df_qrels = df_qrels.drop_duplicates()

    # Merge topics and qrels into a single dataframe
    dataset = (
        df_topics
        .merge(df_qrels, on="topic_id")
        .drop(columns=["topic_id"])
    )

    # Filter out non-english patents
    dataset_en = filter_out_non_english_patents(
        dataset, topics_tfiles_dir, corpus_dir 
    )

    # Add synthetic negative examples
    dataset_en_w_negatives = add_synthetic_negatives(
        dataset_en, corpus_dir, num_negatives_per_positive
    )
    
    return dataset_en_w_negatives    


@click.command()
@click.option(
    "-c", "--corpus", "path_to_corpus",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help=("Directory where the (unzipped) CLEF-IP 2012 corpus is located. "
          "Note that CLEF-IP 2013 uses the same corpus as CLEF-IP 2012.")
)
@click.option(
    "-tq", "--topics-and-qrels-dir", "path_to_topics_and_qrels_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Path to a directory containing the queries (a.k.a. topics)."
)
@click.option(
    "-o", "--output-dir", "path_to_output_dir",
    type=click.Path(file_okay=False),
    required=True,
    help="Directory where the qrels will be saved."
)
def main(path_to_corpus, path_to_topics_and_qrels_dir, path_to_output_dir):
    topics_and_qrels_dir = Path(path_to_topics_and_qrels_dir)
    
    training_topics_and_qrels_dir = (topics_and_qrels_dir
                                     / "clef-ip-2013-clms-psg-training")
    training_topics_tfiles_dir = (training_topics_and_qrels_dir
                                  / "clef-ip-2013-clms-psg-training-tfile")
    training_topics_file = (training_topics_and_qrels_dir
                            / "clef-ip-2013-clms-psg-training-topics.txt")
    training_qrels_file = (training_topics_and_qrels_dir
                           / "clef-ip-2103-clms-psg-training-qrels.txt")
                           # ^ Yes, there's a typo in the filename

    test_topics_tfiles_dir = (topics_and_qrels_dir
                              / "clef-ip-2013-clms-psg-TEST"
                              / "tfiles")
    test_topics_file = (topics_and_qrels_dir
                        / "clef-ip-2013-clms-psg-TEST"
                        / "clef-ip-2013-clms-psg-TEST.txt")
    test_qrels_en_file = (topics_and_qrels_dir
                          / "2013-clef-ip-clsm-to-psg-qrels"
                          # ^ Yes, there's a typo in the directory name
                          / "2013-clef-ip-QRELS-EN-claims-to-passages.txt")

    corpus_dir = Path(path_to_corpus)
    training_dataset = get_qrels(
        corpus_dir,
        training_topics_tfiles_dir,
        training_topics_file,
        training_qrels_file,
        num_negatives_per_positive=1
    )
    test_dataset = get_qrels(
        corpus_dir,
        test_topics_tfiles_dir,
        test_topics_file,
        test_qrels_en_file,
        num_negatives_per_positive=1
    )

    output_dir = Path(path_to_output_dir)
    output_dir.mkdir(parents=True)
    training_dataset.to_csv(output_dir / "train_qrels.csv", index=False)
    test_dataset.to_csv(output_dir / "test_qrels.csv", index=False)

    # Save the patents that appear in the qrels
    train_topic_patents_ucids = training_dataset["patent_ucid"]
    test_topic_patents_ucids = test_dataset["patent_ucid"]
    rel_patents_ucids = pd.concat([
        training_dataset["rel_patent_ucid"], test_dataset["rel_patent_ucid"]
    ]).unique()

    train_topic_patents = [
        training_topics_tfiles_dir / f"{ucid}.xml"
        for ucid in train_topic_patents_ucids
    ]
    test_topic_patents = [
        test_topics_tfiles_dir / f"{ucid}.xml"
        for ucid in test_topic_patents_ucids
    ]
    rel_patents = [
        corpus_dir / get_relative_file_path_from_ucid(ucid)
        for ucid in rel_patents_ucids
    ]
    patents_dir = output_dir / "patents"
    patents_dir.mkdir()

    for patent in train_topic_patents + test_topic_patents + rel_patents:
        shutil.copy(str(patent), str(patents_dir / patent.name))

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
