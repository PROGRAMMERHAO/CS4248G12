---
library_name: transformers
base_model: /mnt/ssd2/Boqian&Zeyu/NLP/zai-tune/checkpoint-8247
tags:
- generated_from_trainer
datasets:
- squad
model-index:
- name: albert
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# albert

This model is a fine-tuned version of [/mnt/ssd2/Boqian&Zeyu/NLP/zai-tune/checkpoint-8247](https://huggingface.co//mnt/ssd2/Boqian&Zeyu/NLP/zai-tune/checkpoint-8247) on the squad dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 12
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 4.47.0.dev0
- Pytorch 2.5.1
- Datasets 3.1.0
- Tokenizers 0.20.3
