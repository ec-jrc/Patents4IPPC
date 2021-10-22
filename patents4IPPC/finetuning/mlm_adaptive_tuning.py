import multiprocessing
import random

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
import datasets
import numpy as np
import torch


def mlm_finetuning(
    model_name_or_path,
    train_texts,
    dev_texts,
    train_batch_size,
    eval_batch_size,
    epochs,
    learning_rate,
    lr_warmup_ratio,
    weight_decay,
    output_dir,
    logging_dir=None,
    max_sequence_length=None,
    do_lower_case=False,
    path_to_previous_checkpoint=None,
    seed=0
):
    """Fine-tune a HuggingFace transformers model on a Masked Language 
    Modeling (MLM) task.

    Args:
        model_name_or_path (str): Name of a pre-trained model hosted on 
          the HuggingFace model hub OR path to a local pre-trained 
          HuggingFace transformers model.
        train_texts (list or array): Collection of texts to use for 
          training.
        dev_texts (list or array): Collection of texts to use for 
          validation.
        train_batch_size (int): Batch size for training.
        eval_batch_size (int): Batch size for validation.
        epochs (int): Number of fine-tuning epochs.
        learning_rate (float): Learning rate to use for fine-tuning.
        lr_warmup_ratio (float): Fraction of fine-tuning iterations 
          during which the learning rate is progressively increased 
          from 0 to `learning_rate`.
        weight_decay (float): Weight decay coefficient to use for 
          regularization.
        output_dir (str): Directory where the fine-tuned model will be 
          saved.
        logging_dir (str, optional): Directory where TensorBoard logs 
          will be saved. Defaults to None, i.e. do not save logs.
        max_sequence_length (int, optional): Maximum sequence length 
          that the Transformer model will be able to handle. Shorter 
          sequences will be padded, whereas longer sequences will be 
          truncated. Defaults to None, i.e. use the pre-trained model's 
          default maximum sequence length.
        do_lower_case (bool, optional): Lowercase the texts before 
          using them for fine-tuning the model. Useful if the model was 
          trained on lowercase text. Defaults to False.
        path_to_previous_checkpoint (str, optional): Path to a previous 
          checkpoint to resume MLM fine-tuning from. Defaults to None, 
          i.e. do not resume from a previous checkpoint.
        seed (int, optional): Random seed used for fine-tuning. 
          Defaults to 0.
    """

    # Fix random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load the model and the tokenizer
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        do_lower_case=do_lower_case
    )
    
    # Load training and validation datasets
    train_dataset = datasets.Dataset.from_dict({'text': train_texts})
    dev_dataset = datasets.Dataset.from_dict({'text': dev_texts})
    
    # Tokenize the samples in the dataset
    max_sequence_length = (max_sequence_length
                           or min(tokenizer.model_max_length,
                                  model.config.max_position_embeddings))    
    def tokenize(samples):
        return tokenizer(
            samples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_sequence_length,
            return_special_tokens_mask=True
        )
    # Disable tqdm bars when Dataset.map(...) is called
    datasets.logging.set_verbosity_error()
    train_dataset_tokenized = train_dataset.map(
        tokenize,
        batched=True,
        num_proc=multiprocessing.cpu_count(),
        remove_columns=['text']
    )
    dev_dataset_tokenized = dev_dataset.map(
        tokenize,
        batched=True,
        num_proc=multiprocessing.cpu_count(),
        remove_columns=['text']
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=len(dev_dataset) > 0,
        evaluation_strategy='epoch',
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=epochs,
        warmup_ratio=lr_warmup_ratio,
        logging_dir=logging_dir,
        logging_strategy='steps' if logging_dir is not None else 'no',
        save_strategy='epoch',
        save_total_limit=epochs,
        seed=seed
        # load_best_model_at_end=True,
        # metric_for_best_model='loss',
        # greater_is_better=False
    )

    # Prepare the Trainer object
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset_tokenized,
        eval_dataset=dev_dataset_tokenized,
        tokenizer=tokenizer
    )
    
    # Start training
    trainer.train(resume_from_checkpoint=path_to_previous_checkpoint)
