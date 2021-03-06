import operator
import random
import math

from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer, losses, models
from torch.utils.data import DataLoader
import numpy as np
import torch


def sentence_transformers_finetuning(
    model_name_or_path,
    train_samples,
    dev_samples,
    loss,
    train_batch_size,
    dev_batch_size,
    max_sequence_length,
    epochs,
    output_path,
    pooling_mode="mean",
    evaluation_steps=1000,
    encoder_attribute_name="encoder",
    learning_rate=2e-5,
    weight_decay=0.01,
    is_sbert_model=False,
    seed=0,
    finetune_top_k_layers=-1
):
    """Fine-tune a HuggingFace transformers or sentence-transformers 
    model on a classification or Semantic Textual Similarity (STS) task.

    Args:
        model_name_or_path (str): Name of a pre-trained model hosted on 
          the HuggingFace model hub / sentence-transformers hub OR path 
          to a local pre-trained HuggingFace transformers OR 
          sentence-transformers model.
        train_samples (iterable): Collection of triplets of the form 
          (text1, text2, label) to use for training.
        dev_samples (iterable): Collection of triplets of the form 
          (text1, text2, label) to use for validation.
        loss (str): Loss function to use for fine-tuning. Pass 'softmax' 
          for classification tasks and 'cosine' for STS tasks.
        train_batch_size (int): Batch size for training.
        dev_batch_size (int): Batch size for validation.
        max_sequence_length (int): Maximum sequence length that the 
          Transformer model will be able to handle. Shorter sequences 
          will be padded, whereas longer sequences will be truncated.
        epochs (int): Number of fine-tuning epochs.
        output_path (str): Path where the fine-tuned model will be saved.
        pooling_mode (str, optional): Pooling strategy to aggregate
          token embeddings into a single sentence embedding. Allowed
          values are "cls", "max" and "mean". Ignored if
          "is_sbert_model" is True (see below). Defaults to "mean".
        evaluation_steps (int, optional): Number of training steps after 
          which the model is evaluated on the validation set. Note that 
          the model is also evaluated after the end of each epoch. 
          Defaults to 1000.
        encoder_attribute_name (str): Name of the attribute that holds 
          the encoder module of the Transformer. Needed in case you want 
          to fine-tune only some of the top layers of the model.        
        learning_rate (float, optional): Learning rate to use for 
          fine-tuning. Defaults to 2e-5.
        weight_decay (float, optional): Weight decay coefficient to use 
          for regularization. Defaults to 0.01.
        is_sbert_model (bool, optional): Whether the specified 
          pre-trained model is a sentence-transformers checkpoint or 
          not. Defaults to False, meaning that it should be interpreted 
          as a plain HuggingFace transformers model.
        seed (int, optional): Random seed used for fine-tuning
          Defaults to 0.
        finetune_top_k_layers (int, optional): Number of top layers to 
          fine-tune while freezing the others. Defaults to -1 (i.e. 
          fine-tune all layers).

    Raises:
        ValueError: `loss` is neither 'softmax' nor 'cosine'.

    Returns:
        SentenceTransformer: The fine-tuned model.
    """

    assert finetune_top_k_layers != 0, \
        ("sentence-transformers doesn't add any additional trainable "
         "parameter on top of the Transformer model, so you must "
         "fine-tune at least one of its layers.")

    # Fix random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if is_sbert_model:
        # Give the SBERT model directly
        model = SentenceTransformer(model_name_or_path)
    else:  # Plain Transformer model
        # Define the Transformer model used for extracting contextual
        # word embeddings
        word_embedding_model = models.Transformer(model_name_or_path)
        # Define the pooling model used for aggregating the output of
        # the contextual word embedding model
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode=pooling_mode
        )
        # Build the model used for fine-tuning, which is the composition 
        # of the contextual word embedding model and the pooling model
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model]
        )

    # Optionally freeze some of the model's layers
    if finetune_top_k_layers > 0:
        transformer_model = model._first_module().auto_model
        encoder = getattr(transformer_model, encoder_attribute_name)
        encoder_layers = next(
            filter(
                lambda module: isinstance(module, torch.nn.ModuleList),
                encoder._modules.values()
            )
        )        
        # Freeze all layers, then unfreeze some
        transformer_model.requires_grad_(False)
        for i in range(1, finetune_top_k_layers + 1):
            encoder_layers[-i].requires_grad_(True)

    # Set the maximum sequence length. Longer sequences are truncated
    model.max_seq_length = max_sequence_length

    # Prepare training data
    train_samples_structured = [
        InputExample(texts=[text1, text2], label=label)
        for (text1, text2, label) in train_samples
    ]
    train_dataloader = DataLoader(
        train_samples_structured, shuffle=True, batch_size=train_batch_size
    )

    # Define loss function
    if loss == 'softmax':
        embedding_dim = model.get_sentence_embedding_dimension()
        unique_labels = set(
            map(operator.attrgetter('label'), train_samples_structured)
        )
        train_loss = losses.SoftmaxLoss(
            model=model,
            sentence_embedding_dimension=embedding_dim,
            num_labels=len(unique_labels)
        )
    elif loss == 'cosine':
        train_loss = losses.CosineSimilarityLoss(model=model)
    else:
        raise ValueError(f'Unknown loss: "{loss}".')

    # Prepare validation data
    dev_samples_structured = [
        InputExample(texts=[text1, text2], label=label)
        for (text1, text2, label) in dev_samples
    ]
    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        dev_samples_structured, batch_size=dev_batch_size, name='dev'
    )

    # Train the model
    warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1)
    # ^ 10% of train data for warm-up
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': learning_rate},
        weight_decay=weight_decay,
        output_path=output_path,
        evaluation_steps=evaluation_steps
    )

    return model
