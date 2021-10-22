from pathlib import Path
import json

from transformers import AutoModel, AutoTokenizer
from tqdm import trange
import numpy as np
import torch

from patents4IPPC.embedders.base_embedder import BaseEmbedder
import utils


class HuggingFaceTransformerEmbedder(BaseEmbedder):

    def __init__(self, model_name_or_path):
        """Text embedder based on a HuggingFace transformers model.

        Args:
            model_name_or_path (str): Name of a pre-trained model 
              hosted on the HuggingFace model hub OR path to a local 
              pre-trained HuggingFace transformers model.
        """

        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    @property
    def embedding_size(self):
        return self.model.config.hidden_size

    def embed_documents(
        self, documents, batch_size=64, do_lowercase=False, show_progress=False
    ):
        texts = (
            documents if not do_lowercase else list(map(str.lower, documents))
        )
        n_documents = len(texts)
        embeddings = []
        for batch_start_idx in trange(
            0, n_documents, batch_size, disable=(not show_progress)
        ):
            batch_end_idx = min(batch_start_idx + batch_size, n_documents)
            # Tokenize the batch of documents
            max_length = min(
                self.tokenizer.model_max_length,
                self.model.config.max_position_embeddings
            )
            inputs = self.tokenizer(
                texts[batch_start_idx:batch_end_idx],
                padding='max_length', # Allows batching
                max_length=max_length,
                truncation=True,
                return_tensors='pt'
            )
            inputs.to(self.device)
            # Run the tokenized batch through the model
            output = self.model(**inputs)
            # Take the average of the output embeddings as our document
            # embeddings
            mean_pooled_output = utils.mean_pool(
                output, inputs['attention_mask']
            )
            embeddings_batch = (
                mean_pooled_output
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            # Add this batch of document embeddings to the list of all
            # embeddings
            embeddings.append(embeddings_batch)
        
        return np.vstack(embeddings)

class DualTransformerEmbedder(BaseEmbedder):
    
    def __init__(self, path_to_pretrained_model_dir):
        """Embedder based on a DualTransformer model. The model used to 
        embed documents is always the query model, including the query 
        mapper layer that sits on top of it.

        Args:
            path_to_pretrained_model_dir (str): Path to a pretrained 
              DualTransformer model.
        """

        pretrained_model_dir = Path(path_to_pretrained_model_dir)        
        # Load the query model, which will be the model that we will use
        # to embed documents (we assume that response embeddings have
        # been precomputed and stored somewhere)
        query_model_location = pretrained_model_dir / 'query_model'
        self.query_model = AutoModel.from_pretrained(str(query_model_location))
        for param in self.query_model.parameters():
            param.requires_grad = False
        self.query_tokenizer = AutoTokenizer.from_pretrained(
            str(query_model_location)
        )

        # Use a GPU if it is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.query_model.to(self.device)

        # We can safely assume that the query mapper exists, because
        # otherwise you'd simply use HuggingFaceTransformerEmbedder
        # instead of DualTransformerEmbedder
        query_mapper_location = pretrained_model_dir / 'query_mapper'
        assert query_mapper_location.exists(), \
               ('The specified pretrained model does not contain a query '
                'mapper. Either use a "HuggingFaceTransformerEmbedder" '
                'instance instead or provide a valid checkpoint (i.e. one '
                'that has a query mapper).')
        mapper_config = json.loads(
            (query_mapper_location / 'mapper_config.json').read_text()
        )
        self.query_mapper = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=mapper_config['query_embedding_size'],
                out_features=mapper_config['hidden_size']
            ),
            torch.nn.LayerNorm((mapper_config['hidden_size'],)),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(
                in_features=mapper_config['hidden_size'],
                out_features=mapper_config['response_embedding_size']
            )
        )
        state_dict = torch.load(str(query_mapper_location / 'mapper.pth'))
        self.query_mapper.load_state_dict(state_dict)
        self.query_mapper.eval()
        self.query_mapper.to(self.device)

    @property
    def embedding_size(self):
        return self.query_model.config.hidden_size

    def embed_documents(
        self, documents, batch_size=64, do_lowercase=False, show_progress=False
    ):
        texts = (
            documents if not do_lowercase else list(map(str.lower, documents))
        )
        n_documents = len(texts)
        embeddings = []
        for batch_start_idx in trange(
            0, n_documents, batch_size, disable=(not show_progress)
        ):
            batch_end_idx = min(batch_start_idx + batch_size, n_documents)
            # Tokenize the batch of documents
            max_length = min(
                self.query_tokenizer.model_max_length,
                self.query_model.config.max_position_embeddings
            )
            inputs = self.query_tokenizer(
                texts[batch_start_idx:batch_end_idx],
                padding='max_length', # Allows batching
                max_length=max_length,
                truncation=True,
                return_tensors='pt'
            )
            inputs.to(self.device)
            # Run the tokenized batch through the model
            output = self.query_model(**inputs)
            # Take the average of the output embeddings as our document
            # embeddings
            mean_pooled_output = utils.mean_pool(
                output, inputs['attention_mask']
            )
            # Project the embeddings using the query mapper
            projected_output = self.query_mapper(mean_pooled_output)
            # Add this batch of document embeddings to the list of all
            # embeddings
            embeddings_batch = (
                projected_output
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            embeddings.append(embeddings_batch)
        
        return np.vstack(embeddings)
