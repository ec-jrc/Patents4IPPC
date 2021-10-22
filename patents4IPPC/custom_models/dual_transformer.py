from pathlib import Path
import operator
import json
import math

from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

from allrank.models.losses import neuralNDCG
import utils


class RankingDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset: pd.DataFrame,
        min_list_length=10,
        precomputed_embeddings_mode=False
    ):
        """Dataset for text ranking tasks. As opposed to normal 
        datasets, a single datapoint in this dataset is actually a 
        collection of <query, response> pairs with different similarity 
        scores. This is necessary as ranking measures such as Spearman 
        rank correlation and NDCG are defined on a collection of 
        samples rather than individual samples.
    
        Args:
            dataset (pd.DataFrame): Dataset with "standard" column 
              names, i.e. it must have at least the following columns: 
              query_id, query, response_id, response, label.
            min_list_length (int, optional): Minimum number of 
              <query, response> pairs in each datapoint. The actual 
              number will be the smallest integer that is 
              >= min_list_length and a multiple of the number of 
              distinct scores in the dataset (the idea is that each 
              datapoint must contain an equal amount of 
              <query, response> pairs for each possible score). 
              Defaults to 10.
            precomputed_embeddings_mode (bool, optional): Whether 
              precomputed response embeddings are available for the 
              subsequent training phase. If True, responses in each 
              datapoint will be represented by their IDs rather than 
              their text body. Defaults to False.
        """

        self.dataset = dataset
        self.unique_labels = list(self.dataset['label'].unique())
        self.min_list_length = min_list_length
        self.n_samples_to_take = math.ceil(
            min_list_length / len(self.unique_labels)
        )
        self.actual_list_length = (self.n_samples_to_take
                                   * len(self.unique_labels))
        self.precomputed_embeddings_mode = precomputed_embeddings_mode                                   

    def __len__(self):
        # NOTE: This is a dummy quantity, i.e. the dataset doesn't
        # actually have this many samples (in fact, it has infinitely
        # many, since each time we take a random sample of the rows).
        # However, since PyTorch requires us to put a fixed number here,
        # we're going to assume that its length is the total number of
        # rows divided by the number of rows that make up a single data
        # point, rounded up
        return math.ceil(len(self.dataset) / self.actual_list_length)

    def __getitem__(self, idx):
        data_point = []
        for l in self.unique_labels:
            # Sample some rows that share the same label. NOTE: we use
            # the `idx` parameter as a random seed so that different
            # accesses to the same index return the same data point
            dataset_fixed_label = self.dataset.loc[self.dataset['label'] == l]
            samples_fixed_label = dataset_fixed_label.sample(
                n=self.n_samples_to_take, random_state=idx
            )
            # Add them to the data point
            if self.precomputed_embeddings_mode:
                relevant_columns = ['query', 'response_id', 'label']
                relevant_columns_dtypes = [str, int, np.float32]
            else:
                relevant_columns = ['query', 'response', 'label']
                relevant_columns_dtypes = [str, str, np.float32]
            data_point.extend(
                samples_fixed_label
                .loc[:, relevant_columns]
                .astype(dict(zip(relevant_columns, relevant_columns_dtypes)))
                .itertuples(index=False, name=None)
            )

        # Shuffle the samples within the data point in order to mix the
        # labels a little bit
        np.random.shuffle(data_point)

        datapoint_dict = {
            'queries': list(map(operator.itemgetter(0), data_point)),
            'responses': list(map(operator.itemgetter(1), data_point)),
            'labels': list(map(operator.itemgetter(2), data_point))
        }
        return datapoint_dict

class DualTransformer(torch.nn.Module):

    def __init__(
        self,
        path_to_pretrained_query_embedder,
        path_to_pretrained_response_embedder=None,
        path_to_pretrained_query_mapper=None,
        max_sentence_length=None,
        precomputed_response_embeddings_size=None,
        freeze_embedders_weights=False,
        include_query_mapper_regardless=False,
        query_mapper_hidden_size=2048
    ):
        """Dual Transformer model for Text Ranking or Semantic Textual 
        Similarity (STS) tasks, possibly consisting of two different 
        Transformer models for embedding queries and responses. If the 
        Transformer model is just one, then the DualTransformer is 
        actually a siamese network architecture.

        Args:
            path_to_pretrained_query_embedder (str): Path to a 
              pretrained HuggingFace transformers model for embedding 
              queries.
            path_to_pretrained_response_embedder (str, optional): Path 
              to a pretrained HuggingFace transformers model for 
              embedding responses. Defaults to None, which means that 
              either the query model is used to embed responses as well 
              OR response embeddings have been precomputed.
            path_to_pretrained_query_mapper (str, optional): Path to a 
              checkpoint of the layer that sits on top of the query 
              model to map query embeddings to the space of response 
              embeddings. Defaults to None, i.e. such layer is not 
              needed because query and response embeddings already have 
              the same dimension. NOTE: ignore this parameter if you 
              are creating a `DualTransformer` instance from outside 
              this class.
            max_sentence_length (int, optional): Maximum sequence 
              length that the two Transformer models can handle. 
              Shorter sequences will be padded, whereas longer 
              sequences will be truncated. Defaults to None, i.e. the 
              default maximum sequence length is used.
            precomputed_response_embeddings_size (int, optional): Size 
              of the precomputed response embeddings, provided that 
              they are available. Defaults to None, i.e. no precomputed 
              response embeddings available.
            freeze_embedders_weights (bool, optional): Whether or not 
              to freeze the weights of the two Transformer models for a 
              subsequent training phase. If True, then there must be a 
              query mapper layer on top of the query model, otherwise 
              there would be no trainable parameters. Defaults to False.
            include_query_mapper_regardless (bool, optional): Include a 
              query mapper on top of the query model even in case query 
              and response models have the same embedding size. 
              Defaults to False.
            query_mapper_hidden_size (int, optional): Number of units 
              in the hidden layer of the query mapper. Defaults to 2048.
        """

        super().__init__()

        # Initialize the query embedder
        self.query_embedder = AutoModel.from_pretrained(
            path_to_pretrained_query_embedder
        )
        self.query_tokenizer = AutoTokenizer.from_pretrained(
            path_to_pretrained_query_embedder
        )
        self.query_embedding_size = self.query_embedder.config.hidden_size
        
        # Optionally freeze weights
        if freeze_embedders_weights:
            for param in self.query_embedder.parameters():
                param.requires_grad = False

        # Optionally initialize the response embedder
        if path_to_pretrained_response_embedder is not None:
            self.response_embedder = AutoModel.from_pretrained(
                path_to_pretrained_response_embedder
            )
            self.response_tokenizer = AutoTokenizer.from_pretrained(
                path_to_pretrained_response_embedder
            )
            self.response_embedding_size = \
                self.response_embedder.config.hidden_size
            
            # Optionally freeze weights
            if freeze_embedders_weights:
                for param in self.response_embedder.parameters():
                    param.requires_grad = False
        else:
            self.precomputed_response_embeddings_size = \
                precomputed_response_embeddings_size
            self.response_embedding_size = \
                (self.precomputed_response_embeddings_size
                 or self.query_embedding_size)

        # Make sure that there will be at least some trainable parameters
        needs_query_mapper = \
            self.query_embedding_size != self.response_embedding_size
        if not needs_query_mapper and not include_query_mapper_regardless:
            assert not freeze_embedders_weights, \
                ('Cannot freeze embedders weights when there is no query '
                 'mapper, otherwise there would be no trainable parameters.')
        # Optionally initialize the fully connected layer for projecting
        # query embeddings to the dimension of response embeddings
        if needs_query_mapper or include_query_mapper_regardless:
            self.query_mapper_hidden_size = query_mapper_hidden_size
            self.query_mapper = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=self.query_embedding_size,
                    out_features=self.query_mapper_hidden_size
                ),
                torch.nn.LayerNorm((self.query_mapper_hidden_size,)),
                torch.nn.Dropout(p=0.1),
                torch.nn.Linear(
                    in_features=self.query_mapper_hidden_size,
                    out_features=self.response_embedding_size
                )
            )
            # Optionally load pre-trained weights for the query mapper.
            # NOTE: Ideally, one should never provide a path to a
            # pretrained query mapper when manually instantiating this
            # class. In fact, only the "from_pretrained()" method below
            # should be allowed to do that
            if path_to_pretrained_query_mapper is not None:
                state_dict = torch.load(path_to_pretrained_query_mapper)
                self.query_mapper.load_state_dict(state_dict)

        # Possibly override the maximum sequence length of the embedders
        self.max_sentence_length = max_sentence_length

        # Initialize the device ("None" refers to the default device)
        self.model_device = None

    @classmethod
    def from_pretrained(
        cls, path_to_pretrained_model, freeze_embedders_weights=False
    ):
        base_path = Path(path_to_pretrained_model)
        # Locate the pretrained query embedder (which must exist,
        # otherwise the checkpoint is not valid)
        path_to_pretrained_query_embedder = str(base_path / 'query_model')
        # Locate the pretrained response embedder and query mapper. Note
        # that these models are not required to exist; in fact, if the
        # response model doesn't exist, it'll default to the query
        # model. On the other hand, if it does exist but its embedding
        # size is the same as the query model's, then we don't need a
        # query mapper
        response_model_location = base_path / 'response_model'
        path_to_pretrained_response_embedder = (
            str(response_model_location)
            if response_model_location.exists()
            else None
        )
        query_mapper_location = base_path / 'query_mapper'
        path_to_pretrained_query_mapper = (
            str(query_mapper_location / 'mapper.pth')
            if query_mapper_location.exists()
            else None
        )
        if query_mapper_location.exists():            
            mapper_config_location = \
                query_mapper_location / 'mapper_config.json'
            mapper_config = json.loads(mapper_config_location.read_text())
            response_embeddings_size = mapper_config['response_embedding_size']
            query_mapper_hidden_size = mapper_config['hidden_size']
        else:
            response_embeddings_size = None
            query_mapper_hidden_size = None
        # Return an instance of the class
        return cls(
            path_to_pretrained_query_embedder,
            path_to_pretrained_response_embedder,
            path_to_pretrained_query_mapper,
            precomputed_response_embeddings_size=response_embeddings_size,
            freeze_embedders_weights=freeze_embedders_weights,
            include_query_mapper_regardless=query_mapper_location.exists(),
            query_mapper_hidden_size=query_mapper_hidden_size
        )

    def to_device(self, device):
        new_self = super().to(device)
        new_self.model_device = device
        return new_self

    def _extract_embeddings(self, ids, response_embeddings):
        embeddings = [
            response_embeddings[id_.item()]
            if isinstance(id_, torch.Tensor)
            else response_embeddings[id_]
            for id_ in ids
        ]
        return torch.tensor(
            np.stack(embeddings),
            dtype=torch.float32,
            device=self.model_device,
            requires_grad=False  # Do not train pre-computed embeddings
        )

    def _forward_single_input(self, x, tokenizer, embedder):
        # Tokenize inputs
        inputs = tokenizer(
            x,
            padding='max_length',
            #max_length=embedder.config.max_position_embeddings if self.max_sentence_length is None else self.max_sentence_length,
            max_length=(self.max_sentence_length
                        or min(tokenizer.model_max_length,
                               embedder.config.max_position_embeddings)),
            truncation=True,
            return_tensors='pt'
        )
        if self.model_device is not None:
            inputs.to(self.model_device)
        # Get the output from the embedder
        output = embedder(**inputs)
        # Perform a mean pooling of the output
        mean_pooled_output = utils.mean_pool(output, inputs['attention_mask'])
        return mean_pooled_output

    def forward(
        self, queries, responses, precomputed_response_embeddings: dict = None
    ):
        if precomputed_response_embeddings is not None:
            assert self.precomputed_response_embeddings_size is not None, \
                   ('If you intend to use pre-computed response embeddings '
                    'while training the model, then you must provide the '
                    'size of such embeddings when instantiating '
                    f'`{type(self).__name__}`.')

        query_embeddings = self._forward_single_input(
            queries,
            self.query_tokenizer,
            self.query_embedder
        )
        if precomputed_response_embeddings is not None:
            response_embeddings = self._extract_embeddings(
                responses, precomputed_response_embeddings
            )
        else:
            response_embeddings = self._forward_single_input(
                responses,
                getattr(self, 'response_tokenizer', self.query_tokenizer),
                getattr(self, 'response_embedder', self.query_embedder)
            )
        # Optionally map query embeddings to the size of response
        # embeddings
        if hasattr(self, 'query_mapper'):
            query_embeddings = self.query_mapper(query_embeddings)

        return query_embeddings, response_embeddings

    def save_pretrained(self, output_path):
        base_output_path = Path(output_path)
        base_output_path.mkdir(parents=True, exist_ok=True)
        # Save the query model in HuggingFace format
        query_model_output_path = str(base_output_path / 'query_model')
        self.query_embedder.save_pretrained(query_model_output_path)
        self.query_tokenizer.save_pretrained(query_model_output_path)

        # If it's not the same as the query model and it's not replaced
        # by pre-computed response embeddings, save the response model
        # in HuggingFace format
        if hasattr(self, 'response_embedder'):
            response_model_output_path = str(
                base_output_path / 'response_model'
            )
            self.response_embedder.save_pretrained(response_model_output_path)
            self.response_tokenizer.save_pretrained(response_model_output_path)

        # If present, save the query mapper too
        if hasattr(self, 'query_mapper'):
            query_mapper_output_path = base_output_path / 'query_mapper'
            query_mapper_output_path.mkdir(parents=True, exist_ok=True)
            mapper_module_output_path = str(
                query_mapper_output_path / 'mapper.pth'
            )
            torch.save(
                self.query_mapper.state_dict(), mapper_module_output_path
            )

            mapper_config = {
                'query_embedding_size': self.query_embedding_size,
                'hidden_size': self.query_mapper_hidden_size,
                'response_embedding_size': self.response_embedding_size
            }
            mapper_config_output_path = \
                query_mapper_output_path / 'mapper_config.json'
            mapper_config_output_path.write_text(
                json.dumps(mapper_config, indent=2)
            )

class NeuralNDCGLoss(torch.nn.Module):

    def __init__(self, cosine_loss_weight=0.0, **kwargs):
        """Implementation of the NeuralNDCG metric 
        (https://arxiv.org/abs/2102.07831) for ranking tasks. Mostly 
        adapted from https://github.com/allegro/allRank.
    
        Args:
            cosine_loss_weight (float, optional): Weight to assign to 
              the cosine part of the loss, i.e. the discrepancy between 
              the cosine similarity of a <query, response> pair and its 
              target label. Defaults to 0, i.e. use the NeuralNDCG loss 
              only.
            **kwargs: Keyword arguments for allrank.models.losses.neuralNDCG()
        """

        super().__init__()

        self.neural_ndcg_kwargs = kwargs
        self.cosine_loss_weight = cosine_loss_weight
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, query_embeddings, response_embeddings, labels):
        predicted_scores = torch.cosine_similarity(
            query_embeddings, response_embeddings, dim=-1
        )

        ndcg_loss = 1 + neuralNDCG(
            y_pred=predicted_scores, y_true=labels, **self.neural_ndcg_kwargs
        )
        cosine_loss = self.mse_loss(predicted_scores, labels)

        final_loss = ((1 - self.cosine_loss_weight) * ndcg_loss
                      + self.cosine_loss_weight * cosine_loss)

        return final_loss

def _process_batch_vectorized(
    batch, model, loss_fn, precomputed_response_embeddings: dict = None
):
    # Prepare inputs and labels
    # (the variables below all have shape (batch_size, list_length))
    queries = np.transpose(batch['queries'])
    if precomputed_response_embeddings is not None:
        responses = torch.stack(batch['responses']).T
    else:
        responses = np.transpose(batch['responses'])
    labels = torch.stack(batch['labels']).float().T # pylint: disable=no-member
    if model.model_device is not None:
        labels = labels.to(model.model_device)

    # Flatten queries/responses, then compute their embeddings
    queries_flat = queries.reshape((-1,)).tolist()
    responses_flat = responses.reshape((-1,)).tolist()
    query_embeddings_flat, response_embeddings_flat = model(
        queries_flat, responses_flat, precomputed_response_embeddings
    )
    # Reshape query/response embeddings, then compute the loss
    batch_size, list_length = queries.shape
    query_embeddings = query_embeddings_flat.reshape((
        batch_size, list_length, -1 # Last dimension is embedding_size
    ))
    response_embeddings = response_embeddings_flat.reshape((
        batch_size, list_length, -1 # Last dimension is embedding_size
    ))
    return loss_fn(query_embeddings, response_embeddings, labels)

def _process_batch_iterative(
    batch, model, loss_fn, precomputed_response_embeddings: dict = None
):
    # NOTE: batch['queries'] and batch['labels'] are lists of tuples,
    # and their shape is (list_length, batch_size)
    query_embeddings_batches = []
    response_embeddings_batches = []
    for queries_batch, responses_batch in zip(
        batch['queries'], batch['responses']
    ):
        # Compute embeddings for a batch of queries and responses. The
        # resulting output shape will be (batch_size, embedding_size)
        query_embeddings_batch, response_embeddings_batch = model(
            list(queries_batch),
            list(responses_batch),
            precomputed_response_embeddings
        )

        query_embeddings_batches.append(query_embeddings_batch)
        response_embeddings_batches.append(response_embeddings_batch)

    # Stack batches together to obtain embedding tensors of shape
    # (batch_size, list_length, embedding_size)
    query_embeddings = torch.stack(query_embeddings_batches, dim=1)
    response_embeddings = torch.stack(response_embeddings_batches, dim=1)

    # Reshape labels to have shape (batch_size, list_length), then
    # compute and return the loss
    labels = torch.stack(batch['labels']).float().T
    if model.model_device is not None:
        labels = labels.to(model.model_device)
    return loss_fn(query_embeddings, response_embeddings, labels)

def _train_loop(
    dataloader,
    model,
    loss_fn,
    optimizer,
    iterative_mode=False,
    precomputed_response_embeddings: dict = None
):
    model.train() # Set the model to training mode
    process_batch = (_process_batch_iterative
                     if iterative_mode
                     else _process_batch_vectorized)
    progress_bar = tqdm(dataloader, leave=True)
    for batch in progress_bar:
        # Compute the loss for the current batch
        loss = process_batch(
            batch, model, loss_fn, precomputed_response_embeddings
        )
        # Backpropagate gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Display the loss
        progress_bar.set_postfix_str(f'Loss: {loss.item():.4f}')

@torch.no_grad()
def _eval(
    dataloader,
    model,
    loss_fn,
    iterative_mode=False,
    precomputed_response_embeddings: dict = None
):
    model.eval() # Set the model to evaluation mode
    process_batch = (_process_batch_iterative
                     if iterative_mode
                     else _process_batch_vectorized)
    return np.mean([
        process_batch(
            batch, model, loss_fn, precomputed_response_embeddings
        ).item()
        for batch in tqdm(dataloader, leave=False)
    ])

def train_dual_transformer(
    dual_transformer_model,
    train_dataloader,
    loss_fn,
    optimizer,
    epochs,
    eval_dataloader=None,
    run_evaluation_on_training_set=False,
    iterative_mode=False,
    precomputed_response_embeddings: dict = None
):
    """Train a DualTransformer model on a Text Ranking task.

    Args:
        dual_transformer_model (DualTransformer): The DualTransformer 
          model to fine-tune.
        train_dataloader (torch.utils.data.DataLoader): DataLoader 
          providing training batches.
        loss_fn (torch.nn.Module): Loss function to optimize during 
          training.
        optimizer (torch.optim.Optimizer): `Optimizer` instance 
          representing the optimization algorithm to use for training.
        epochs (int): Number of fine-tuning epochs.
        eval_dataloader (torch.utils.data.DataLoader, optional): 
          DataLoader providing validation batches. Defaults to None, 
          i.e. skip validation.
        run_evaluation_on_training_set (bool, optional): Whether to run 
          evaluation on the whole training set at the end of each epoch 
          or not. The upside is that you get a better estimate of the 
          training error, but the downside is that the training
          procedure will be slowed down. Defaults to False.
        iterative_mode (bool, optional): Process elements of a batch in 
          an iterative fashion rather than a vectorized fashion. Allows 
          to save memory at the expense of a little overhead in 
          computation. Defaults to False.
        precomputed_response_embeddings (dict, optional): Dictionary 
          containing precomputed response embeddings. Keys are integer 
          IDs of the responses, and the corresponding values are NumPy 
          arrays representing text embeddings for each that response. 
          Allows to save memory avoiding to load the response model.
          Defaults to None, i.e. do not use precomputed response 
          embeddings.

    Returns:
        tuple(list, list): training and validation loss values for each 
          epoch.
    """

    train_loss_history = []
    valid_loss_history = []
    for t in range(epochs):
        print(f'Epoch {t+1}/{epochs}')
        _train_loop(
            train_dataloader,
            dual_transformer_model,
            loss_fn,
            optimizer,
            iterative_mode,
            precomputed_response_embeddings
        )
        if run_evaluation_on_training_set:
            train_loss = _eval(
                train_dataloader,
                dual_transformer_model,
                loss_fn,
                iterative_mode,
                precomputed_response_embeddings
            )
            train_loss_history.append(train_loss)
            print(f'Training loss at the end of the epoch: {train_loss:.4f}')
        if eval_dataloader is not None:
            eval_loss = _eval(
                eval_dataloader,
                dual_transformer_model,
                loss_fn,
                iterative_mode,
                precomputed_response_embeddings
            )
            valid_loss_history.append(eval_loss)
            print(f'Validation loss at the end of the epoch: {eval_loss:.4f}')

    return train_loss_history, valid_loss_history
