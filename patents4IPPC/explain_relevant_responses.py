from pathlib import Path

import click
import pandas as pd

from patents4IPPC.explainability import (
    HierarchicalTransformerTextSimilarityExplainer,
    HuggingFaceTextSimilarityExplainer
)


def format_hf_explainer_output(
    explainer_output, output_dir, query_id, response_id
):
    (
        query_tokens,
        query_attributions,
        response_tokens,
        response_attributions
    ) = explainer_output

    output_subdir = Path(output_dir) / f"q_{query_id}_r_{response_id}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    query_attributions = pd.DataFrame({
        "token": query_tokens,
        "attribution_score": query_attributions
    })
    query_attributions.to_csv(
        str(output_subdir / f"q_{query_id}_tokens.csv"), index=False
    )

    response_attributions = pd.DataFrame({
        "token": response_tokens,
        "attribution_score": response_attributions
    })
    response_attributions.to_csv(
        str(output_subdir / f"r_{response_id}_tokens.csv"), index=False
    )    

def format_ht_explainer_output(
    explainer_output, output_dir, query_id, response_id
):
    (
        query_tokens,
        query_token_attributions,
        query_segments,
        query_segment_attributions,
        response_tokens,
        response_token_attributions,
        response_segments,
        response_segment_attributions,        
    ) = explainer_output

    output_subdir = Path(output_dir) / f"q_{query_id}_r_{response_id}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    n_segments_in_query = len(query_segments)
    for i in range(n_segments_in_query):
        segment_tokens_and_attributions = pd.DataFrame({
            "token": query_tokens[i],
            "attribution_score": query_token_attributions[i]
        })
        segment_tokens_and_attributions.to_csv(
            str(output_subdir / f"q_{query_id}_segment_{i:03d}.csv"),
            index=False
        )
    query_segments_and_attributions = pd.DataFrame({
        "segment": query_segments,
        "attribution_scores": query_segment_attributions
    })
    query_segments_and_attributions.to_csv(
        str(output_subdir / f"q_{query_id}_segments.csv"), index=False
    )

    n_segments_in_response = len(response_segments)
    for i in range(n_segments_in_response):
        segment_tokens_and_attributions = pd.DataFrame({
            "token": response_tokens[i],
            "attribution_score": response_token_attributions[i]
        })
        segment_tokens_and_attributions.to_csv(
            str(output_subdir / f"r_{response_id}_segment_{i:03d}.csv"),
            index=False
        )        
    response_segments_and_attributions = pd.DataFrame({
        "segment": response_segments,
        "attribution_scores": response_segment_attributions
    })
    response_segments_and_attributions.to_csv(
        str(output_subdir / f"r_{response_id}_segments.csv"), index=False
    )    

@click.command()
@click.option(
    "-mc", "--model-checkpoint", "path_to_model",
    type=click.Path(exists=True),
    required=True,
    help="Path to a pre-trained model whose predictions you want to explain."
)
@click.option(
    "-mt", "--model-type",
    type=click.Choice(["huggingface", "hierarchical"]),
    required=True,
    help="Type of the pre-trained model."
)
@click.option(
    "-p", "--pooling-mode",
    type=click.Choice(["cls", "max", "mean"]),
    default=None,
    help=("Pooling strategy to transform token embeddings into sentence "
          "embeddings. Only required when --model-type is \"huggingface\". "
          "Note that when --model-type is \"hierarchical\", the pooling "
          "strategy for the segment Transformer is automatically extracted "
          "from the configuration files contained in the model checkpoint.")
)
@click.option(
    "-pr", "--predictions", "path_to_predictions",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help=("Path to a .csv file containing predictions. The file should have "
          "at least the following columns: query, query_id, response, "
          "response_id, similarity_score.")
)
@click.option(
    "-s", "--steps-for-integrated-gradients",
    type=int,
    default=50,
    help="Number of steps for approximating integrated gradients."
)
@click.option(
    "-o", "--output-dir",
    type=click.Path(file_okay=False),
    required=True,
    help="Path top a directory where explanations will be saved."
)
def main(
    path_to_model,
    model_type,
    pooling_mode,
    path_to_predictions,
    steps_for_integrated_gradients,
    output_dir
):
    if model_type == "huggingface":
        assert pooling_mode is not None, \
            ("You must provide a --pooling-mode when using a "
             "\"huggingface\" model.")
        explainer = HuggingFaceTextSimilarityExplainer(
            path_to_model, pooling_mode=pooling_mode
        )
    elif model_type == "hierarchical":
        explainer = HierarchicalTransformerTextSimilarityExplainer(
            path_to_model,
            disable_gradients_computation_for_segment_transformer=True
        )

    predictions = pd.read_csv(path_to_predictions)
    for _, row in predictions.iterrows():
        explainer_output = explainer.explain(
            row["query"],
            row["response"],
            n_steps=steps_for_integrated_gradients,
            internal_batch_size=2,
            normalize_attributions=False
        )
        if model_type == "huggingface":
            format_hf_explainer_output(
                explainer_output,
                output_dir,
                row["query_id"],
                row["response_id"]
            )
        elif model_type == "hierarchical":
            format_ht_explainer_output(
                explainer_output,
                output_dir,
                row["query_id"],
                row["response_id"]                
            )

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
