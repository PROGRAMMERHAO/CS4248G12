import os
import torch
import json
import collections
from tqdm import tqdm
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import csv
import subprocess
import sys

# Define the path to your fine-tuned model and tokenizer
model_path = "../models/final_model"
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available

# Load the model and tokenizer from the local directory
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create the pipeline with the locally loaded model and tokenizer
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)

# Define a directory to save the final model and tokenizer (optional)
final_save_directory = '../models/albert_model'
os.makedirs(final_save_directory, exist_ok=True)

# Save the final model (in case you want to save to another directory)
model.save_pretrained(final_save_directory)
tokenizer.save_pretrained(final_save_directory)

print(f'Model and tokenizer saved to {final_save_directory}')

# Load the dev data
with open('../data/dev-v1.1.json', 'r') as f:
    squad_data = json.load(f)

dev_data = []

for article in squad_data['data']:
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            qid = qa['id']
            dev_data.append({'qid': qid, 'question': question, 'context': context})

# Ensure the eval directory exists
eval_directory = '../eval'
os.makedirs(eval_directory, exist_ok=True)

# Function to split context into chunks
def split_context(context, max_len=400, stride=100):
    tokens = context.split()
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = " ".join(tokens[i:i + max_len])
        chunks.append(chunk)
        if i + max_len >= len(tokens):
            break
    return chunks

# Prepare the grid search parameters for a 5x5 grid
max_len_values = [128, 256, 320, 384, 512]  # Expanded range for max_len
stride_values = [64, 128, 192, 256, 320]    # Expanded range for stride

# Open CSV file in append mode
csv_file = 'grid_search_results.csv'

# Write header only once if the file is new
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as csvfile:
        fieldnames = ['max_len', 'stride', 'exact_match', 'f1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

# Loop over the grid search parameters
for max_len in max_len_values:
    for stride in stride_values:
        print(f'Running prediction with max_len={max_len}, stride={stride}')
        # Prediction code with context splitting and saving predictions to JSON
        predictions = collections.OrderedDict()

        for item in tqdm(dev_data):
            context_chunks = split_context(item['context'], max_len=max_len, stride=stride)
            best_answer = ""
            highest_score = 0

            # Run the QA pipeline on each context chunk
            for chunk in context_chunks:
                result = qa_pipeline(question=item['question'], context=chunk)
                if result['score'] > highest_score:
                    best_answer = result['answer']
                    highest_score = result['score']

            predictions[item['qid']] = best_answer

        # Save predictions to a new JSON file for better organization
        prediction_file = os.path.join(eval_directory, f'albert_predictions_maxlen{max_len}_stride{stride}.json')
        with open(prediction_file, 'w') as f:
            json.dump(predictions, f)

        print(f'Predictions saved to {prediction_file}')

        # Run evaluation script and capture output
        evaluation_script = '../eval/evaluate-v2.0.py'
        dev_data_file = '../data/dev-v1.1.json'

        # Prepare the command
        cmd = [sys.executable, evaluation_script, dev_data_file, prediction_file]

        try:
            # Run the evaluation script and capture the output
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            output = result.stdout
            # Parse the output JSON
            metrics = json.loads(output)
            exact_match = metrics['exact']
            f1 = metrics['f1']

            # Write results to CSV after each evaluation
            with open(csv_file, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['max_len', 'stride', 'exact_match', 'f1'])
                writer.writerow({'max_len': max_len, 'stride': stride, 'exact_match': exact_match, 'f1': f1})
            print(f'Results: EM={exact_match}, F1={f1}')

        except subprocess.CalledProcessError as e:
            print(f"Error during evaluation: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")

print(f'Grid search results saved to {csv_file}')
