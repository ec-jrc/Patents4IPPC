from dataclasses import dataclass, asdict
from pathlib import Path
import json

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from .utils import concat_encoded_inputs


@dataclass
class TrainingArguments:
    learning_rate_for_document_embedder: float = 1e-2
    learning_rate_for_segment_transformer: float = 5e-5
    weight_decay: float = 0
    batch_size: int = 2
    eval_batch_size: int = 2
    seed: int = 0
    freeze_segment_embedder_weights: bool = False

    @classmethod
    def from_json(cls, path_to_json_file):
        with open(path_to_json_file, "r") as f:
            config_params = json.load(f)
        return cls(**config_params)

class DocumentSimilarityTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        training_arguments,
        eval_dataset=None,
        output_dir=None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.training_arguments = training_arguments
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir) if output_dir is not None else None

        self.train_loss_history = []
        self.eval_loss_history = []
        self.best_train_loss = np.inf
        self.best_eval_loss = np.inf

        self._initialize_loss_and_optimizer()
        self._initialize_data_loaders()

    def _initialize_loss_and_optimizer(self):
        self.loss_fn = torch.nn.CosineEmbeddingLoss(margin=0, reduction="mean")

        trainable_parameters = [{
            "params": self.model.document_embedder.parameters(),
            "lr": self.training_arguments.learning_rate_for_document_embedder,
            "weight_decay": self.training_arguments.weight_decay
        }]
        if self.training_arguments.freeze_segment_embedder_weights:
            for tensor in self.model.segment_transformer.parameters():
                tensor.requires_grad = False
        else:
            trainable_parameters.append({
                "params": self.model.segment_transformer.parameters(),
                "lr": self.training_arguments.learning_rate_for_segment_transformer,
                "weight_decay": self.training_arguments.weight_decay            
            })
        self.optimizer = torch.optim.Adam(trainable_parameters)

    def _initialize_data_loaders(self):
        torch.manual_seed(self.training_arguments.seed)
        def collate_fn(list_of_samples):
            LEFT_ENCODED_SEGMENTS = 0
            LEFT_DOCUMENT_IDS = 1
            RIGHT_ENCODED_SEGMENTS = 2
            RIGHT_DOCUMENT_IDS = 3
            LABELS = 4
            collated_samples = [
                concat_encoded_inputs([
                    sample[LEFT_ENCODED_SEGMENTS] for sample in list_of_samples
                ]),
                np.concatenate([
                    sample[LEFT_DOCUMENT_IDS] for sample in list_of_samples
                ]),
                concat_encoded_inputs([
                    sample[RIGHT_ENCODED_SEGMENTS] for sample in list_of_samples
                ]),
                np.concatenate([
                    sample[RIGHT_DOCUMENT_IDS] for sample in list_of_samples
                ]),
                torch.tensor([
                    sample[LABELS] for sample in list_of_samples
                ])
            ]
            return collated_samples

        self.data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.training_arguments.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True
        )
        if self._is_evaluation_needed():
            self.eval_data_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.training_arguments.eval_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=True
            )        

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
        eval_loss = np.mean([
            self._compute_loss(batch).item()
            for batch in tqdm(
                self.eval_data_loader, desc="Eval loss computation", leave=False
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
        loss = self._compute_loss(batch)
        self._adjust_learning_rate_based_on_total_number_of_segments(batch)
        self._backpropagate_gradients(loss)
        self._restore_learning_rate()
        return loss.item()

    def _compute_loss(self, batch):
        # Make sense of the batch
        left_encoded_segments_batch, left_document_ids_batch = batch[0:2]
        right_encoded_segments_batch, right_document_ids_batch = batch[2:4]
        labels = batch[-1]
        
        # Get document embeddings
        left_document_embeddings_batch = self.model(
            left_encoded_segments_batch, left_document_ids_batch
        )
        right_document_embeddings_batch = self.model(
            right_encoded_segments_batch, right_document_ids_batch
        )

        return self.loss_fn(
            left_document_embeddings_batch,
            right_document_embeddings_batch,
            labels
        )

    def _adjust_learning_rate_based_on_total_number_of_segments(self, batch):
        if self.training_arguments.freeze_segment_embedder_weights:
            return
        
        base_learning_rate_of_segment_transformer = \
            self.optimizer.param_groups[1]["lr"]
        
        n_left_segments = len(batch[0])
        n_right_segments = len(batch[2])
        rescaling_factor = (n_left_segments + n_right_segments) / 32 # TODO: find a better solution than hardcoding
        
        adjusted_learning_rate_of_segment_transformer = \
            base_learning_rate_of_segment_transformer * rescaling_factor
        self.optimizer.param_groups[1]["lr"] = \
            adjusted_learning_rate_of_segment_transformer

    def _backpropagate_gradients(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()        

    def _restore_learning_rate(self):
        if self.training_arguments.freeze_segment_embedder_weights:
            return
                
        self.optimizer.param_groups[1]["lr"] = \
            self.training_arguments.learning_rate_for_segment_transformer

    def _save_model_if_better_than_previous_ones(self):
        if len(self.eval_loss_history) == 0:
            if self.train_loss_history[-1] == self.best_train_loss:
                self._save_model_and_tokenizer()
        elif self.eval_loss_history[-1] == self.best_eval_loss:
            self._save_model_and_tokenizer()

    def _save_model_and_tokenizer(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model_output_dir = self.output_dir / "model"
        self.model.save_pretrained(model_output_dir)
        self.train_dataset.tokenizer.save_pretrained(
            model_output_dir / "segment_transformer"
        )
