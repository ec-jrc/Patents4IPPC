from pathlib import Path
import csv

import click
import pandas as pd

from clef_ip_utils import extract_content_from_patent


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
