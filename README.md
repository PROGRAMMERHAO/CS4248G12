# CS4248G12 Project

# NLP Model Evaluation and Fine-tuning Project

This project is dedicated to evaluating, fine-tuning, and experimenting with multiple NLP models, including ALBERT, Llama, and BERT. The repository contains datasets, model checkpoints, prediction outputs, and development scripts to facilitate NLP question answering task on the SQuAD dataset.

## Table of Contents

- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training and Fine-tuning](#model-training-and-fine-tuning)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Development Scripts](#development-scripts)


---

## Project Overview

This repository includes tools for fine-tuning, and evaluating NLP models, with a focus on performance optimization. Our model SplaXBERT is obtained from the fine-tuning of ALBERT-xlarge with mixed precision training and context-splitting. This project enables model comparison through cross-attention techniques, highway layer addition, and other fine-tuning strategies.

## Folder Structure
```
NLP Project
├── data/
│   ├── dev-v1.1.json
│   ├── dev-v2.0.json
│   ├── train-v1.1.json
│   └── train-v2.0.json
│
├── eval/
│   ├── albert_predictions.json
│   ├── albert_refine_predictions.json
│   ├── distilled_predictions.json
│   ├── fine_tuned_albert_base_predictions.json
│   ├── fine_tuned_albert_predictions.json
│   ├── fine_tuned_predictions_albert_new.json
│   ├── fine_tuned_predictions_albert.json
│   ├── fine_tuned_predictions_bert.json
│   └── SplaxBERT_prediction.json
│
├── models/
│   ├── fine_tuned_albert_base_model/
│   ├── fine_tuned_albert_model/
│   ├── fine_tuned_albert_with_cross_attention_model/
│   ├── fine_tuned_albert_xlarge_mix_2_model/
│   ├── fine_tuned_albert_xlarge_model/
│   ├── fine_tuned_albert_xlarge_nomix_model/
│   ├── fine_tuned_albert_xlarge_rly_mix_model/
│   ├── fine_tuned_model_bert/
│   └── SplaxBERT_model/
│
├── scripts/                # Development scripts
│   ├── alternative studies/
│   │   ├── fine_tuned_albert_with_cross_attention.py
│   │   ├── fine_tuned_albert_with_highway.py
│   │   ├── hybert_architecture.py
│   │   └── voting.py
│   ├── albert_inference_pipeline.py
│   ├── context_split_grid_search.py
│   ├── fine_tuned_albert_base_qa.py
│   ├── fine_tuned_albert_xlarge_wo_mix.py
│   ├── fine_tuned_bert.py
│   ├── grid_search_results.csv
│   ├── langchain_qa.py
│   ├── llama_qa.py
│   ├── prediction.py
│   ├── student_teacher_bert_qa.py
│   └── student_teacher_qa.py
│
├── src/                    # Actual production models
│   ├── albert_inference_pipeline_with_context_splitting.py
│   └── fine_tuned_albert_xlarge_with_mix.py
│
├── utils/
│
├── .gitignore
├── README.md
└── requirements.txt

```

### disclaimer
The file structure has been changed for the purpose of submission, please check the path before running the scripts.

### `data/`
Contains the datasets used for model training and evaluation.
- `dev-v1.1.json`, `dev-v2.0.json`: Development sets for model testing.
- `train-v1.1.json`, `train-v2.0.json`: Training datasets.

### `eval/`
Holds prediction outputs and evaluation results from various models.
We have done a lot of predictions and the names are self-explanatory.
- `evaluate-v2.0.py`: Evaluation script for the exact match and F1 scores on the dataset.

### `models/`
We have tested a lot of models, here are some sample models shown due to size constraints. Stores fine-tuned model checkpoints for various versions and configurations.
- `fine_tuned_albert_base_model`: Base version of the fine-tuned ALBERT model.
- `fine_tuned_albert_with_cross_attention_model`: Fine-tuned ALBERT model with cross-attention configuration.
- `fine_tuned_bert_model`: Fine-tuned BERT model.
- `fine_tuned_albert_xlarge_wo_mix_model`: Fine-tuned ALBERT-xlarge model.
-  `SplaxBERT_model`: Fine-tuned SplaxBERT model.

### `src/`
Contains the actual model code used for production and inference.
- `albert_inference_pipeline_with_context_splitting.py`: Main pipeline for inference with context-splitting functionality.
- `fine_tuned_albert_xlarge_mix.py`: Code for fine-tuning ALBERT-xlarge model with mixed precision.

### `scripts/`
Contains development scripts used during the experimentation and fine-tuning process.
- `albert_inference_pipeline.py`: Pipeline for running inference using the ALBERT model (development version).
- `context_split_grid_search.py`: Script to perform grid search on different context spliting configurations.
- `fine_tuned_albert_base_qa.py`: Script for fine-tuning the base ALBERT model.
- `fine_tuned_albert_xlarge_wo_mix.py`: Script for fine-tuning the ALBERT model without mixing.
- `fine_tuned_bert_qa.py`: Script for fine-tuning the BERT model.
- `student_teacher_bert.py`, `student_teacher_qa.py`: Scripts for implementing student-teacher training techniques.
- `langchain_qa.py`: Script for direct prompting based on Llama.
- `llama_qa.py`: Script for promtp chaining with few-shot prompting based on Llama.

#### `scripts/alternative studies/`
Contains scripts for alternative model studies and experiments.
- `fine_tuned_albert_with_cross_attention.py`: Fine-tuning ALBERT model with cross-attention.
- `fine_tuned_albert_with_highway.py`: Fine-tuning ALBERT model with highway layers.

### `utils/`
Utility functions to support various operations across scripts and model code.
- `albert_prediction.py`: Utility functions for generating predictions using the pretrained ALBERT-base model.
- `albert_with_highway_precision.py`: Utility functions for generating predictions using the pretrained ALBERT-base model with highway layer.
- `bert_prediction.py`: Utility functions for generating predictions using the pretrained BERT model.
- `test_cuda.py`: Utility functions for testing CUDA availability.
- `test_model_loading.py`: Utility functions for testing model loading.
- `text_to_json.py`: Utility functions for converting text to JSON format.
---

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
cd CS4248G12
pip install -r requirements.txt
```

Ensure you have the necessary environment setup, including Python and any required libraries as specified in `requirements.txt`.

## Usage

### Data Preparation

Place your training and evaluation datasets in the `data/` folder. Update paths in the relevant scripts if necessary.

### Model Training and Fine-tuning

Use scripts in the `src/` folder to run the main models for production or experimental pipelines. For example:

```bash
cd src
python fine_tuned_albert_xlarge_with_mix.py
```

This script will fine-tune the ALBERT-xlarge model with mixed precision. It will also run prediction on testing dataset and save the results in the eval folder.

### Evaluation

To evaluate the model, navigate to the `src` directory first, then run the evaluation script:

```bash
python eval/evaluate-v2.0.py data/dev-v1.1.json eval/fine_tuned_predictions.json
```

### Inference

Run inference on your dataset by navigating to the `src` directory and using the production pipeline:

```bash
cd src
python albert_inference_pipeline_with_context_splitting.py
```

## Development Scripts

The `scripts/` folder contains additional development scripts for experimentation, testing, and fine-tuning.

To fine tune a model, use:

```bash
cd scripts
python fine_tuned_albert_base_qa.py
```
Other scripts have similar usage.

