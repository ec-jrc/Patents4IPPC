from transformers import BertTokenizerFast

from patents4IPPC.embedders.baseline import tfidf, glove, use
from patents4IPPC.embedders.advanced import huggingface, hierarchical
from patents4IPPC import preprocessing


def get_embedder(
    model_type, path_to_model_checkpoint=None, pooling_mode="mean"
):
    if model_type == "tfidf":
        if path_to_model_checkpoint is not None:
            embedder = tfidf.TfidfEmbedder.from_pretrained(
                path_to_model_checkpoint
            )
        else:
            embedder = tfidf.TfidfEmbedder(
                return_dense_embeddings=True,
                lowercase=True, # To increase the variety of possible words
                preprocessor=preprocessing.preprocess_for_tfidf,
                tokenizer=BertTokenizerFast.from_pretrained(
                    "bert-base-uncased"
                ).tokenize,
                max_df=0.7,
                max_features=8192 # TODO: Is this the right number?
            )
    elif model_type == "glove":
        embedder = glove.GloveEmbedder(path_to_model_checkpoint)
    elif model_type == "use":
        embedder = use.UniversalSentenceEncoderEmbedder(
            path_to_model_checkpoint
        )
    elif model_type == "huggingface":
        embedder = huggingface.HuggingFaceTransformerEmbedder(
            path_to_model_checkpoint, pooling_mode=pooling_mode
        )
    elif model_type == "dual":
        embedder = huggingface.DualTransformerEmbedder(
            path_to_model_checkpoint, pooling_mode=pooling_mode
        )
    elif model_type == "hierarchical":
        embedder = hierarchical.HierarchicalTransformerEmbedder(
            path_to_model_checkpoint
        )
    else:
        raise ValueError(f'Unknown model type "{model_type}".')

    return embedder
