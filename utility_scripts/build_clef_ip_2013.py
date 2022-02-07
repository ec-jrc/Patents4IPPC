from pathlib import Path
import csv
import re
import xml.etree.ElementTree as ET

import click
import pandas as pd


btext = ["b", "i", "o", "u", "sub", "sup", "smallcaps"]
ptext = ["b", "i", "o", "u", "sub", "sup", "smallcaps", "br", "dl", "ul", "ol", "sl"]
supported_children = {
    "abstract": ["abst-problem", "abst-solution", "heading", "p"],
    "abst-problem": ["p"],
    "abst-solution": ["p"],
    "heading": [*btext],
    "p": [*ptext],
    "claim": ["claim-text"],
    "claim-text": [*ptext, "claim-text"],
    "b":         [     "i", "o", "u", "sub", "sup", "smallcaps"],
    "i":         ["b",      "o", "u", "sub", "sup", "smallcaps"],
    "o":         ["b", "i",      "u", "sub", "sup", "smallcaps"],
    "u":         ["b", "i", "o",      "sub", "sup", "smallcaps"],
    "sub":       ["b", "i", "o", "u", "sub", "sup", "smallcaps", "sub2", "sup2"],
    "sup":       ["b", "i", "o", "u", "sub",        "smallcaps", "sub2", "sup2"],
    "smallcaps": ["b", "i", "o", "u", "sub", "sup",              "sub2", "sup2"],
    "sub2":      ["b", "i", "o", "u", "sub", "sup", "smallcaps",         "sup2"],
    "sup2":      ["b", "i", "o", "u", "sub", "sup", "smallcaps", "sub2"        ],
    "br": [],
    "dl": ["dt", "dd"],
    "ul": ["li"],
    "ol": ["li"],
    "sl": ["li"],
    "dt": [*btext],
    "dd": [*ptext],
    "li": [*ptext]
}
requires_whitespace = {
    "abstract": True,
    "abst-problem": True,
    "abst-solution": True,
    "heading": True,
    "p": True,    
    "claim": True,
    "claim-text": True,
    "b": False,
    "i": False,
    "o": False,
    "u": False,
    "sub": False,
    "sup": False,
    "smallcaps": False,
    "sub2": False,
    "sup2": False,
    "br": True,
    "dl": True,
    "ul": True,
    "ol": True,
    "sl": True,
    "dt": True,
    "dd": True,
    "li": True
}

def get_text_from_tag(tag: ET.Element):
    final_text = tag.text or ""
    for child in tag:            
        if child.tag not in supported_children[tag.tag]:
            if child.tail is not None:
                final_text += " " + child.tail
            continue
        
        child_text = get_text_from_tag(child)
        
        if requires_whitespace[child.tag] and final_text != "":
            final_text += " "
        
        final_text += child_text
        
        if child.tail is not None:
            if requires_whitespace[child.tag]:
                final_text += " "
            final_text += child.tail

    final_text = re.sub(r"\s+", " ", final_text)

    return final_text.strip()

def extract_content_from_patent(
    patent_file: Path,
    add_section_tags=False,
    abstract_only=False
):
    xml_document_root = ET.parse(patent_file)
    
    abstract = xml_document_root.find(".//abstract[@lang='EN']")
    abstract_text = get_text_from_tag(abstract) if abstract is not None else ""
    if add_section_tags:
        abstract_text = f"[abstract] {abstract_text}"    

    if abstract_only:
        return abstract_text, []

    claims = xml_document_root.findall(".//claims[@lang='EN']/claim")   
    if patent_file.stem.startswith("EP"):
        # ^ EPO patent
        claims_texts = [get_text_from_tag(claim) for claim in claims]
    else:
        # ^ WO patent
        assert len(claims) == 1  
        # ^ WO patents contain only one <claim> tag that contains all of
        # the claims
        single_claim_text = get_text_from_tag(claims[0])
        raw_claims_texts = re.sub(
            r"\s+(\d{1,2}\.\s+[^\d])",  # e.g. " 1. T"
            lambda match: "\n" + match.groups()[0],
            single_claim_text
        ).split("\n")
        claims_texts = [
            re.sub(r"^\d{1,2}\.\s*", "", raw_text)
            for raw_text in raw_claims_texts
            if re.search(r"^\d{1,2}\.\s*", raw_text)
        ]
    
    if add_section_tags:
        claims_texts = [f"[claim] {claim_text}" for claim_text in claims_texts]

    return abstract_text, claims_texts

def make_dataset(
    subset,
    qrels,
    qrels_dir,
    output_dir,
    as_csv_files,
    add_section_tags,
    abstract_only
):
    for _, row in qrels.iterrows():
        q_patent_ucid = row["patent_ucid"]
        q_patent_file = qrels_dir / "patents" / f"{q_patent_ucid}.xml"        
        
        rel_patent_ucid = row["rel_patent_ucid"]
        rel_patent_file = qrels_dir / "patents" / f"{rel_patent_ucid}.xml"        

        label = row["label"]

        q_patent_abstract, q_patent_claims = extract_content_from_patent(
            q_patent_file, add_section_tags, abstract_only
        )
        rel_patent_abstract, rel_patent_claims = extract_content_from_patent(
            rel_patent_file, add_section_tags, abstract_only
        )        

        if as_csv_files:
            qrels_output_path = output_dir / f"{subset}.csv"
            if not qrels_output_path.exists():
                qrels_output_path.write_text(
                    "query_id,query,response_id,response,label\n"
                )
            with open(qrels_output_path, mode="a") as fp:
                q_patent_fused_content = " ".join(
                    [q_patent_abstract] + q_patent_claims
                )
                rel_patent_fused_content = " ".join(
                    [rel_patent_abstract] + rel_patent_claims
                )

                writer = csv.writer(fp, quoting=csv.QUOTE_ALL)
                writer.writerow([
                    q_patent_ucid,
                    q_patent_fused_content,
                    rel_patent_ucid,
                    rel_patent_fused_content,
                    str(label)
                ])
        else:
            q_patent_output_path = \
                output_dir / subset / "qs" / f"{q_patent_ucid}.dat"
            rel_patent_output_path = \
                output_dir / subset / "rels" / f"{rel_patent_ucid}.dat"
            qrels_output_path = output_dir / subset / "qrels.txt"

            q_patent_output_path.parent.mkdir(parents=True, exist_ok=True)
            rel_patent_output_path.parent.mkdir(parents=True, exist_ok=True)
            qrels_output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(q_patent_output_path, "w") as fp:
                fp.writelines("\n".join([q_patent_abstract] + q_patent_claims))
            with open(rel_patent_output_path, "w") as fp:
                fp.writelines("\n".join([rel_patent_abstract] + rel_patent_claims))
            with open(qrels_output_path, "a+") as fp:                
                fp.write(f"qs/{q_patent_ucid}.dat,rels/{rel_patent_ucid}.dat,{label}\n")


@click.command()
@click.option(
    "-q", "--qrels", "path_to_qrels_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Directory where the qrels are located."
)
@click.option(
    "-o", "--output-dir", "path_to_output_dir",
    type=click.Path(file_okay=False),
    required=True,
    help="Directory where the dataset will be saved."
)
@click.option(
    "-f", "--as-csv-files",
    type=bool,
    is_flag=True,
    help=("Save train and test dataset as CSV files with \"standard\" column "
          "names. This will be convenient for training models other than a "
          "Hierarchical Transformer.")
)
@click.option(
    "-s", "--add-section-tags",
    type=bool,
    is_flag=True,
    help=("Whether to add section tags based on which section a given piece "
          "of text was taken from. Useful for BERT for Patents.")
)
@click.option(
    "-a", "--abstract-only",
    type=bool,
    is_flag=True,
    help="Extract only the patents' abstracts, ignoring the claims."
)
def main(
    path_to_qrels_dir,
    path_to_output_dir,
    as_csv_files,
    add_section_tags,
    abstract_only
):
    qrels_dir = Path(path_to_qrels_dir)
    train_qrels = pd.read_csv(str(qrels_dir / "train_qrels.csv"))
    test_qrels = pd.read_csv(str(qrels_dir / "test_qrels.csv"))

    output_dir = Path(path_to_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    make_dataset(
        subset="train",
        qrels=train_qrels,
        qrels_dir=qrels_dir,
        output_dir=output_dir,
        as_csv_files=as_csv_files,
        add_section_tags=add_section_tags,
        abstract_only=abstract_only
    )

    make_dataset(
        subset="test",
        qrels=test_qrels,
        qrels_dir=qrels_dir,
        output_dir=output_dir,
        as_csv_files=as_csv_files,
        add_section_tags=add_section_tags,
        abstract_only=abstract_only
    )

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
