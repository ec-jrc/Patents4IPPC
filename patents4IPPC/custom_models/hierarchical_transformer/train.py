from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from .utils import (
    concat_encoded_inputs, index_encoded_inputs, move_encoded_inputs_to_device
)
from utils import unique_values_in_order_of_appearance


@dataclass
class TrainingArguments:
    learning_rate_for_document_embedder: float = 1e-2
    learning_rate_for_segment_transformer: float = 5e-5
    weight_decay: float = 0.0
    batch_size: int = 8
    cosine_loss_margin: float = 0.4
    seed: int = 0
    top_layers_to_train: int = 0

    @classmethod
    def from_json(cls, path_to_json_file):
        with open(path_to_json_file, "r") as f:
            config_params = json.load(f)
        return cls(**config_params)

class DocumentSimilarityTrainer:
    def __init__(
        self,
        model,
        segment_transformer_encoder_attr_name,
        train_dataset,
        training_arguments,
        eval_dataset=None,
        output_dir=None,
        cache_dir="cached_segment_embeddings"
    ):
        self.model = model
        self.segment_transformer_encoder_attr_name = \
            segment_transformer_encoder_attr_name
        self.train_dataset = train_dataset
        self.training_arguments = training_arguments
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.cache_dir = Path(cache_dir)

        if training_arguments.top_layers_to_train == 0:
            self._create_cache_dirs_for_segment_embeddings()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.train_loss_history = []
        self.eval_loss_history = []
        self.best_train_loss = np.inf
        self.best_eval_loss = np.inf

        self.loss_fn = torch.nn.CosineEmbeddingLoss(
            margin=training_arguments.cosine_loss_margin, reduction="mean"
        )
        self._initialize_optimizer()
        self._initialize_data_loaders()

    def _create_cache_dirs_for_segment_embeddings(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir_train = self.cache_dir / "train"
        self.cache_dir_train.mkdir(exist_ok=True)
        if self._is_evaluation_needed():
            self.cache_dir_eval = self.cache_dir / "eval"
            self.cache_dir_eval.mkdir(exist_ok=True)

    def _initialize_optimizer(self):
        document_embedder_params_spec = {
            "params": self.model.document_embedder.parameters(),
            "lr": self.training_arguments.learning_rate_for_document_embedder,
            "weight_decay": self.training_arguments.weight_decay
        }

        segment_transformer_params_specs = \
            self._get_segment_transformer_params_specs()

        self.optimizer = torch.optim.Adam([
            document_embedder_params_spec, *segment_transformer_params_specs
        ])

    def _initialize_data_loaders(self):
        torch.manual_seed(self.training_arguments.seed)
        def collate_fn(list_of_samples):
            LEFT_ENCODED_SEGMENTS = 0
            LEFT_DOCUMENT_ID_AND_N_SEGMENTS = 1
            RIGHT_ENCODED_SEGMENTS = 2
            RIGHT_DOCUMENT_ID_AND_N_SEGMENTS = 3
            LABELS = 4
            collated_samples = [
                move_encoded_inputs_to_device(
                    concat_encoded_inputs([
                        sample[LEFT_ENCODED_SEGMENTS]
                        for sample in list_of_samples
                    ]),
                    self.device
                ),
                [
                    sample[LEFT_DOCUMENT_ID_AND_N_SEGMENTS]
                    for sample in list_of_samples
                ],
                move_encoded_inputs_to_device(
                    concat_encoded_inputs([
                        sample[RIGHT_ENCODED_SEGMENTS]
                        for sample in list_of_samples
                    ]),
                    self.device
                ),
                [
                    sample[RIGHT_DOCUMENT_ID_AND_N_SEGMENTS]
                    for sample in list_of_samples
                ],
                torch.tensor([
                    sample[LABELS] for sample in list_of_samples
                ], device=self.device)
            ]
            return collated_samples

        self.data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.training_arguments.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        if self._is_evaluation_needed():
            self.eval_data_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.training_arguments.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )        

    def _get_segment_transformer_params_specs(self):
        segment_transformer_layers = self._get_segment_transformer_layers()
        segment_transformer_params_specs = \
            self._make_top_layers_of_segment_transformer_trainable(
                segment_transformer_layers
            )
        return segment_transformer_params_specs        

    def _get_segment_transformer_layers(self):
        segment_transformer_encoder = getattr(
            self.model.segment_transformer,
            self.segment_transformer_encoder_attr_name
        )
        segment_transformer_encoder_layers = next(
            filter(
                lambda v: isinstance(v, torch.nn.ModuleList),
                segment_transformer_encoder._modules.values()
            )
        )
        segment_transformer_layers = [
            self.model.segment_transformer.embeddings,
            *segment_transformer_encoder_layers
        ]
        return segment_transformer_layers

    def _make_top_layers_of_segment_transformer_trainable(
        self, segment_transformer_layers
    ):
        n_layers = len(segment_transformer_layers)
        assert self.training_arguments.top_layers_to_train <= n_layers, \
            ("You requested to train the top "
                f"{self.training_arguments.top_layers_to_train} layers, but "
                f"the model only has {n_layers} layers ({n_layers - 1} "
                f"encoder layers + 1 embedding layer.")

        if self.training_arguments.top_layers_to_train < 0:
            self.training_arguments.top_layers_to_train = n_layers

        params_specs = []
        for i, layer in enumerate(reversed(segment_transformer_layers), 1):
            should_layer_be_trained = i <= self.training_arguments.top_layers_to_train
            for p in layer.parameters():
                p.requires_grad = should_layer_be_trained                
            if should_layer_be_trained:
                params_specs.append({
                    "params": layer.parameters(),
                    "lr": self.training_arguments.learning_rate_for_segment_transformer,
                    "weight_decay": self.training_arguments.weight_decay                        
                })
        
        return params_specs

    def _is_evaluation_needed(self):
        return self.eval_dataset is not None

    def train(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")
            train_loss = self._run_single_epoch_of_training()
            self._update_train_loss_history(train_loss)
            print(f"{'Train loss:':<11} {train_loss:.4f}")
            if self._is_evaluation_needed():
                eval_loss = self._eval()
                self._update_eval_loss_history(eval_loss)
                print(f"{'Eval loss:':<11} {eval_loss:.4f}")
            
            self._maybe_save_checkpoint()
        
        if self.output_dir is not None:
            self._save_training_arguments()
            self._save_loss_history()

    def _run_single_epoch_of_training(self):
        self.model.train()  # Set model to training mode
        self.model.to(self.device)  # Possibly move the model to the GPU
        progress_bar = tqdm(self.data_loader, leave=True)
        batch_losses = []
        for batch in progress_bar:
            loss_value = self._run_single_training_step(batch)
            progress_bar.set_postfix_str(f"Loss: {loss_value:.4f}")
            batch_losses.append(loss_value)
        
        train_loss = np.mean(batch_losses)
        return train_loss

    def _update_train_loss_history(self, train_loss):
        self.train_loss_history.append(train_loss)
        self.best_train_loss = min(self.best_train_loss, train_loss)

    @torch.no_grad()
    def _eval(self):
        self.model.eval()  # Set model to evaluation mode
        cache_dir = getattr(self, "cache_dir_eval", None)       
        eval_loss = np.mean([
            self._compute_loss(batch, cache_dir).item()
            for batch in tqdm(
                self.eval_data_loader,
                desc="Eval loss computation",
                leave=False
            )
        ])
        return eval_loss

    def _update_eval_loss_history(self, eval_loss):
        self.eval_loss_history.append(eval_loss)
        self.best_eval_loss = min(self.best_eval_loss, eval_loss)

    def _maybe_save_checkpoint(self):
        if self.output_dir is None:
            return
        self._save_model_if_better_than_previous_ones()

    def _save_training_arguments(self):       
        training_arguments_file = self.output_dir / "training_arguments.json"
        training_arguments_file.write_text(
            json.dumps(
                asdict(self.training_arguments),
                indent=2,
                sort_keys=True
            ) + "\n"
        )

    def _save_loss_history(self):
        history = pd.DataFrame({
            "epoch": list(range(1, 1 + len(self.train_loss_history))),
            "train_loss": self.train_loss_history,
            "eval_loss": self.eval_loss_history or None
        })
        history.to_csv(str(self.output_dir / "loss_history.csv"), index=False)

    def _run_single_training_step(self, batch):
        cache_dir = getattr(self, "cache_dir_train", None)
        loss = self._compute_loss(batch, cache_dir)
        self._adjust_learning_rate_based_on_total_number_of_segments(batch)
        self._backpropagate_gradients(loss)
        self._restore_learning_rate()
        return loss.item()

    def _compute_loss(self, batch, cache_dir=None):
        left_encoded_segments_batch = batch[0]
        left_documents_ids_and_n_segments_batch = batch[1]
        right_encoded_segments_batch = batch[2]
        right_documents_ids_and_n_segments_batch = batch[3]
        labels = batch[4]
        
        left_document_embeddings_batch, right_document_embeddings_batch = \
            self._get_document_embeddings(
                left_encoded_segments_batch,
                left_documents_ids_and_n_segments_batch,
                right_encoded_segments_batch,
                right_documents_ids_and_n_segments_batch,
                cache_dir
            )

        return self.loss_fn(
            left_document_embeddings_batch,
            right_document_embeddings_batch,
            labels
        )

    def _adjust_learning_rate_based_on_total_number_of_segments(self, batch):
        n_left_segments = len(batch[0])
        n_right_segments = len(batch[2])
        rescaling_factor = ((n_left_segments + n_right_segments)
                            / self.training_arguments.batch_size)
        
        base_lr_of_segment_transformer = \
            self.training_arguments.learning_rate_for_segment_transformer
        adjusted_lr_of_segment_transformer = \
            base_lr_of_segment_transformer * rescaling_factor
        
        for param_group in self.optimizer.param_groups[1:]:
            param_group["lr"] = adjusted_lr_of_segment_transformer

    def _backpropagate_gradients(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()        

    def _restore_learning_rate(self):
        for param_group in self.optimizer.param_groups[1:]:
            param_group["lr"] = \
                self.training_arguments.learning_rate_for_segment_transformer

    def _get_document_embeddings(
        self,
        left_encoded_segments,
        left_documents_ids_and_n_segments,
        right_encoded_segments,
        right_documents_ids_and_n_segments,
        cache_dir        
    ):
        left_segment_embeddings = self._get_segment_embeddings(
            left_encoded_segments,
            left_documents_ids_and_n_segments,
            cache_dir
        )

        right_segment_embeddings = self._get_segment_embeddings(
            right_encoded_segments,
            right_documents_ids_and_n_segments,
            cache_dir
        )

        left_document_embeddings = self.model.get_document_embeddings(
            left_segment_embeddings, left_documents_ids_and_n_segments
        )

        right_document_embeddings = self.model.get_document_embeddings(
            right_segment_embeddings, right_documents_ids_and_n_segments
        )

        return left_document_embeddings, right_document_embeddings

    def _get_segment_embeddings(
        self, encoded_segments, document_ids_and_n_segments, cache_dir=None
    ):
        if cache_dir is None:
            return self.model.get_segment_embeddings(encoded_segments)          
        
        cached_segment_embeddings = self._get_cached_segment_embeddings(
            document_ids_and_n_segments, cache_dir
        )

        non_cached_segment_embeddings = \
            self._get_non_cached_segment_embeddings_then_cache_them(
                encoded_segments, document_ids_and_n_segments, cache_dir
            )
        
        all_segment_embeddings = {
            **cached_segment_embeddings,
            **non_cached_segment_embeddings
        }
        return torch.concat([
            all_segment_embeddings[doc_id]
            for doc_id, _ in document_ids_and_n_segments
        ])

    def _get_cached_segment_embeddings(
        self, document_ids_and_n_segments, cache_dir
    ):
        def is_document_cached(document_id):
            return (cache_dir is not None
                    and (cache_dir / f"{document_id}.npy").exists())        
        
        cached_documents_ids_and_n_segments = filter(
            lambda tup: is_document_cached(tup[0]),
            document_ids_and_n_segments
        )
        # Load cached segment embeddings
        cached_segment_embeddings = self._load_cached_segment_embeddings(
            cache_dir, cached_documents_ids_and_n_segments
        )

        return cached_segment_embeddings

    def _get_non_cached_segment_embeddings_then_cache_them(
        self, encoded_segments, document_ids_and_n_segments, cache_dir
    ):
        def is_document_cached(document_id):
            return (cache_dir is not None
                    and (cache_dir / f"{document_id}.npy").exists())        
        
        cached_segments_mask = np.array([
            is_document_cached(doc_id)
            for doc_id, n_segments in document_ids_and_n_segments
            for _ in range(n_segments)
        ])
        if all(cached_segments_mask):
            # No need to compute anything because all segment embeddings 
            # were previously cached
            non_cached_segment_embeddings = {}
        else:
            # Compute segment embeddings for non-cached documents
            encoded_segments_of_non_cached_documents = index_encoded_inputs(
                encoded_segments, ~cached_segments_mask
            )
            non_cached_segment_embeddings_stacked = \
                self.model.get_segment_embeddings(
                    encoded_segments_of_non_cached_documents
                )
            # Organize them in a dictionary
            non_cached_documents_ids_and_n_segments = filter(
                lambda tup: not is_document_cached(tup[0]),
                document_ids_and_n_segments
            )
            non_cached_segment_embeddings = {}
            last_index = 0
            for doc_id, n_segments in non_cached_documents_ids_and_n_segments:
                non_cached_segment_embeddings[doc_id] = \
                    non_cached_segment_embeddings_stacked[
                        last_index:last_index+n_segments
                    ]
                last_index += n_segments
            # Cache them
            self._cache_segment_embeddings(
                cache_dir, non_cached_segment_embeddings
            )
        
        return non_cached_segment_embeddings

    def _save_model_if_better_than_previous_ones(self):
        if len(self.eval_loss_history) == 0:
            if self.train_loss_history[-1] == self.best_train_loss:
                self._save_model_and_tokenizer()
        elif self.eval_loss_history[-1] == self.best_eval_loss:
            self._save_model_and_tokenizer()

    def _load_cached_segment_embeddings(
        self, cache_dir, cached_documents_ids_and_n_segments
    ):
        cached_segment_embeddings = {
            document_id: torch.tensor(
                np.load(str(cache_dir / f"{document_id}.npy")),
                device=self.device
            )
            for document_id, _ in cached_documents_ids_and_n_segments
        }
        return cached_segment_embeddings

    def _cache_segment_embeddings(self, cache_dir, segment_embeddings):
        for document_id, tensor in segment_embeddings.items():
            output_path = cache_dir / f"{document_id}.npy"
            np.save(str(output_path), tensor.detach().cpu().numpy())

    def _save_model_and_tokenizer(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model_output_dir = self.output_dir / "model"
        self.model.save_pretrained(model_output_dir)
        self.train_dataset.tokenizer.save_pretrained(
            model_output_dir / "segment_transformer"
        )
