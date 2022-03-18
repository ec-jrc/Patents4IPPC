from pathlib import Path

import numpy as np
import joblib
import click

from patents4IPPC.embedders.utils import get_embedder
from patents4IPPC.similarity_search.faiss_ import index_documents_using_faiss
from clef_ip_utils import extract_content_from_patent


@click.command()
@click.option(
    "-c", "--corpus-dir", "path_to_corpus_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Directory containing a CLEF-IP corpus."
)
@click.option(
    "-f", "--files-list", "path_to_files_list",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=("Text file containing the list of files present in the corpus. Each "
          "line should contain a single file path, where said path must be "
          "relative to the corpus directory. If not provided, the list of "
          "files will be obtained automatically, though it may take a long "
          "time due to the extremely intricate directory structure of a "
          "CLEF-IP corpus.")
)
@click.option(
    "-mt", "--model-type",
    type=click.Choice([
        "tfidf", "glove", "use", "huggingface", "dual", "hierarchical"
    ]),
    required=True,
    help="Type of model to use for indexing the corpus."
)
@click.option(
    "-mc", "--model-checkpoint", "path_to_model_checkpoint",
    type=click.Path(exists=True),
    default=None,
    help=("Path to a pre-trained model. Required unless \"--model-type\" is "
          "\"tfidf\", in which case there are two possibilities: either this "
          "parameter is provided, meaning that a pre-trained TF-IDF model is "
          "used to index the corpus, or it is not, meaning that a fresh "
          "TF-IDF model is fitted on the corpus, then it is used to index it.")
)
@click.option(
    "-p", "--pooling-mode",
    type=click.Choice(["cls", "max", "mean"]),
    default=None,
    help=("Pooling strategy for aggregating token embeddings into sentence "
          "embeddings. Required only when \"--model-type\" is \"huggingface\" "
          "or \"dual\".")
)
@click.option(
    "-b", "--batch-size",
    type=int,
    default=2,
    help="Number of documents to encode at once."
)
@click.option(
    "-o", "--output-path",
    type=click.Path(),
    required=True,
    help=("Location where the FAISS index representing the CLEF-IP corpus "
          "will be saved.")
)
@click.option(
    "--tfidf-output-path",
    type=click.Path(),
    default=None,
    help=("Location where the fitted TF-IDF model will be saved. Required "
          "when \"--model-type\" is \"tfidf\" and \"--model-checkpoint\" was "
          "not specified.")
)
def main(
    path_to_corpus_dir,
    path_to_files_list,
    model_type,
    path_to_model_checkpoint,
    pooling_mode,
    batch_size,
    output_path,
    tfidf_output_path
):
    if model_type != "tfidf":
        assert path_to_model_checkpoint is not None, \
               "Please provide a model checkpoint."
    if model_type == "tfidf" and path_to_model_checkpoint is None:
        assert tfidf_output_path is not None, \
               ("Please provide a path where the fitted TF-IDF model will be "
                "saved.")
    if model_type in ["huggingface", "dual"]:
        assert pooling_mode is not None, \
            f"You must provide \"--pooling-mode\" for model of type {model_type}."

    embedder = get_embedder(
        model_type, path_to_model_checkpoint, pooling_mode=pooling_mode
    )

    # Get list of files in corpus
    path_to_corpus_dir = Path(path_to_corpus_dir)
    if path_to_files_list is not None:
        with open(path_to_files_list, "r") as fp:
            relative_paths = fp.read().strip().split("\n")
        corpus_files = [path_to_corpus_dir / p for p in relative_paths]
    else:
        corpus_files = list(path_to_corpus_dir.glob("**/*.xml"))

    # Extract the content of each patent
    patents = []
    for patent_file in corpus_files:
        abstract, claims = extract_content_from_patent(patent_file)
        if model_type == "hierarchical":
            patent_content = "[SEGMENT_SEP]".join([abstract] + claims)
        else:
            patent_content = " ".join([abstract] + claims)
        patents.append(patent_content)

    # Optionally fit the embedder
    if model_type == "tfidf" and path_to_model_checkpoint is None:
        embedder.fit(np.array(patents))
        joblib.dump(embedder, tfidf_output_path)

    # Embed the patents and create a FAISS index
    index_documents_using_faiss(
        documents=patents,
        ids=np.arange(len(patents)),
        embedder=embedder,
        batch_size=batch_size,
        store_on_disk=True,
        filename=output_path
    )
    if path_to_files_list is None:
        output_path_for_files_list = Path(output_path).with_suffix(".filelist")
        output_path_for_files_list.write_text(
            str(file_.relative_to(path_to_corpus_dir)) + "\n"
            for file_ in corpus_files
        )

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
