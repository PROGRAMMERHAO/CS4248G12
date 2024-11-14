# CS4248G12 Project

This project sets up a question-answering system using knowledge distillation to fine-tune a smaller model based on a larger pre-trained model's outputs. It includes data processing, model training, and evaluation.

## Setup Guide

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/PROGRAMMERHAO/CS4248G12.git
cd CS4248G12
```

### 2. Set Up Virtual Environment

Create and activate a virtual environment to manage dependencies:

```bash
python -m venv env
source env/bin/activate       # On MacOS/Linux
.\env\Scripts\activate        # On Windows
```

### 3. Install Required Packages

Install the necessary dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Configure CUDA

If you have a CUDA-compatible GPU, set it up for faster training:

1. Navigate to the `scripts` folder:

   ```bash
   cd scripts
   ```

2. Run `cuda.py` to verify CUDA setup:

   ```bash
   python cuda.py
   ```

   Ensure CUDA is detected. This step enables training on GPU if available.

### 5. Create Training Scripts

Prepare your training scripts within the `scripts` folder. For this project, the training scripts are already included. The main script, `student_teacher_qa.py`, trains a student model using knowledge distillation from a larger teacher model.

For training and evaluation, you can either run or create your own scripts following the format of `fine_tuned_qa.py` or `student_teacher_qa.py`. These scripts provide structured setups for both fine-tuning and knowledge distillation processes and include saving the model and predictions in appropriate directories.

### 6. Train the Model

Run `student_teacher_qa.py` to train the model. This script will save the fine-tuned model into a new folder within `models`:

```bash
python scripts/student_teacher_qa.py
```

This will train the student model using the knowledge distillation approach and save the output model to `models/distilled_model`.

Run `fine_tuned_albert_xlarge_mix.py` to train the model with mixed precision. This script will save the fine-tuned model into a new folder within `models`:

```bash
python scripts/fine_tuned_albert_xlarge_mix.py
```

Run `fine_tuned_albert_xlarge_nomix.py` to train the model with mixed precision. This script will save the fine-tuned model into a new folder within `models`:

```bash
python scripts/fine_tuned_albert_xlarge_mix.py
```

### 7. Run Predictions

Although the training scripts will run the predictions automatically, if you want to use our pretrained models:

1. Make sure you have the relevant model folder under `models`.
2. Change the model you want to load in `scripts/albert_prediction.py` where tokenizer and model are initialized. For example, to load ALBERT-xlarge model pretrained without mixed precision,
```
tokenizer = AlbertTokenizerFast.from_pretrained('../models/fine_tuned_albert_xlarge_nomix_model')
model = AlbertForQuestionAnswering.from_pretrained('../models/fine_tuned_albert_xlarge_nomix_model')
```

Run `scripts/albert_prediction.py` to load the models and carry out predictions

```bash
python scripts/albert_prediction.py
```



### 8. Run Evaluation

To evaluate the fine-tuned model, use the `evaluate-v2.0.py` script in the `eval` folder:

1. Ensure that `student_teacher_qa.py` (or any other script from the training section) has saved predictions to `eval/fine_tuned_predictions.json`.
2. Run the following command to evaluate the model’s performance:

   ```bash
   python eval/evaluate-v2.0.py data/dev-v1.1.json eval/fine_tuned_predictions.json
   ```

This command will compute evaluation metrics on the development set (`dev-v1.1.json`) and output the results to the console.

## Project Structure

- **data/**: Contains training and development data files.
- **env/**: Virtual environment directory (generated after setup).
- **models/**: Directory for saving trained models. Models are saved in subfolders like `models/distilled_model`.
- **scripts/**: Contains all scripts, including training and CUDA setup scripts.
  - `cuda.py`: Checks CUDA availability.
  - `student_teacher_qa.py`: Main script for training with knowledge distillation.
  - `fine_tuned_bert.py`: Main script for training BERT without mixed precision.
  - `fine_tuned_albert_xlarge_nomix.py`: Main script for training ALBERT-xlarge without mixed precision.
  - `fine_tuned_albert_xlarge_mix.py`: Main script for training ALBERT-xlarge with mixed precision.
  - `albert_prediction`: Script for running predictions on dataset using pretrained models.
  - `langchain_qa.py`: Script for ChatBot-based QA prediction.
- **eval/**: Folder for evaluation scripts and results.
  - `evaluate-v2.0.py`: Script to evaluate predictions against ground truth.
- **tools/**: Folder for supplementary tools.
  - `text_to_json.py`: Script to convert predictions saved in txt format into json format for evaluation purpose.
- **requirements.txt**: Lists all required Python packages.

## Notes

- Make sure to run scripts in the appropriate directories as specified in each step.
- For best performance, run on a machine with a CUDA-compatible GPU.
- The `student_teacher_qa.py` script will create the necessary directories if they don’t already exist, so no manual folder creation is necessary.

## License

This project is developed for educational purposes in CS4248 at NUS. Please adhere to your institution's guidelines for code usage and submission policies.
