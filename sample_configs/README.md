# Sample configuration files

This directory contains sample configuration files needed for each of the `finetune_*.py` scripts inside the `utility_scripts` directory. The values chosen for each parameter therein are just examples, meaning that you may want to adapt them to your specific needs. What follows is a brief description of the meaning of such parameters.

## Common parameters

- `epochs`: Number of fine-tuning epochs

- `learning_rate`: Learning rate of the optimization algorithm

- `weight_decay`: Weight decay term

- `max_sequence_length`: Maximum sequence length (in terms of tokens). Longer sequences will be truncated

## File-specific parameters

- `config_mlm.json`
  
   - `train_batch_size`: Batch size for training
  
   - `eval_batch_size`: Batch size for evaluation
  
   - `lr_warmup_ratio`: Ratio of total training steps used for a linear warmup from 0 to `learning_rate`

- `config_sbert.json`
  
   - `train_batch_size`: Batch size for training
  
   - `dev_batch_size`: Batch size for evaluation
  
   - `seed`: Random seed (for reproducibility)

- `config_dual_transformer.json`
  
   - `query_mapper_hidden_size`:  Since query and response embeddings might have different dimensions, DualTransformer has a so-called *query mapper* on top of the query model whose job is to project the query embedding into the space of the response embedding. Before that, however, the query embedding is projected into an intermediate space whose dimension is `query_mapper_hidden_size`
  
   - `min_list_length`: As of now, DualTransformer can only be trained with the NeuralNDCG loss, which is a differentiable approximation of NDCG. Given how NDCG is defined, a *single* training example has to be a list of <query,response> pairs. `min_list_length` is the minimum length of such list. The reason why it is not its *exact* length is because we make sure that each list contains the same number of pairs for each possible label. For example, if the dataset contains <query,response> pairs whose labels denote their similarity score in a scale from 0 to 2, then a *single* training example for DualTransformer will be a list containing at least `min_list_length` pairs whose labels are equally distributed among 0, 1 and 2. Concretely, if `min_list_length=10`, the list will actually contain 12 <query,response> pairs (4 for each label)
  
   - `batch_size`: Batch size for both training and evaluation
  
   - `cosine_loss_weight`: Weight assigned to the term of the loss function that measures the discrepancy between the cosine similarity of two sequence embeddings and their target label. The other term measures the NeuralNDCG loss. Must be between 0 and 1
