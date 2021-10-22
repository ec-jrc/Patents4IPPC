import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics import ndcg_score


def compute_cosine_scores(
    dataset,
    mdl,
    batch_size=2,
    precomputed_response_embeddings=None,
    do_lowercase=False,
    fraction=1.0
):
    if fraction != 1.0:
        dataset = dataset.sample(frac=fraction)
    
    queries = dataset['query'].values.tolist()
    query_embeddings = mdl.embed_documents(
        queries,
        batch_size=batch_size,
        do_lowercase=do_lowercase,
        show_progress=True
    )
    if precomputed_response_embeddings is None:
        responses = dataset['response'].values.tolist()
        response_embeddings = mdl.embed_documents(
            responses, batch_size=batch_size, do_lowercase=do_lowercase
        )
    else:
        response_embeddings = np.stack([
            precomputed_response_embeddings[id_]
            for id_ in dataset['response_id']
        ]).astype(np.float32)
    
    cosine_scores = 1 - paired_cosine_distances(
        query_embeddings, response_embeddings
    )

    result = dataset[['query_id', 'response_id', 'label']]
    result['cosine_score'] = cosine_scores
    return result

def compute_spearman(df_cosine_scores):
    eval_spearman_cosine, _ = spearmanr(
        df_cosine_scores['label'], df_cosine_scores['cosine_score']
    )
    return eval_spearman_cosine

def compute_spearman_querywise(df_cosine_scores, do_print=False):
    def metric_fn(labels, cosine_scores):
        return spearmanr(labels, cosine_scores).correlation
    return compute_metric(df_cosine_scores, metric_fn, 'Spearman', do_print)

def compute_ndcg(df_cosine_scores, do_print=False):
    def metric_fn(labels, cosine_scores):
        return ndcg_score(
            y_true=labels.reshape(1, -1), y_score=cosine_scores.reshape(1, -1)
        )
    return compute_metric(df_cosine_scores, metric_fn, 'NDCG', do_print)

def compute_metric(df_cosine_scores, metric_fn, metric_name, do_print=False):
    # loop only on unique id
    unique_ids = set(df_cosine_scores['query_id'].values)

    # list of metric values for each id
    metric_values = []

    for id_ in unique_ids:
        # filter only rows with given id
        df_fixed_id = df_cosine_scores[df_cosine_scores['query_id'] == id_]

        # If all candidate abstracts for a given query have the same
        # label, NDCG and Spearman will have meaningless values. Since
        # we do have such degenerate examples in GS1, we don't want our
        # average metric value to be affected by them
        if len(df_fixed_id['label'].unique()) == 1:
            continue

        labels = df_fixed_id['label'].values
        cosine_scores = df_fixed_id['cosine_score'].values

        metric_value = metric_fn(labels, cosine_scores)
        metric_values.append(metric_value)
        if do_print:
            print(f'\t{metric_name} on query {id_}: {metric_value}')

    return np.mean(metric_values), np.std(metric_values)    
