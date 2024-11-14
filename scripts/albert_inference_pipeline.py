import os
import torch
import json
import collections
from tqdm import tqdm
from transformers import pipeline

# Load the model and tokenizer
model_name = "ktrapeznikov/albert-xlarge-v2-squad-v2"
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available

qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name, device=device)

# Define a directory to save the final model and tokenizer
final_save_directory = '../models/albert_model'
os.makedirs(final_save_directory, exist_ok=True)

# Get model and tokenizer from the pipeline
model = qa_pipeline.model
tokenizer = qa_pipeline.tokenizer

# Save the final model
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

# Prediction code and saving predi ctions to JSON
predictions = collections.OrderedDict()

for item in tqdm(dev_data):
    result = qa_pipeline(question=item['question'], context=item['context'])
    predictions[item['qid']] = result['answer']

# Save predictions to a JSON file
with open(os.path.join(eval_directory, 'albert_predictions.json'), 'w') as f:
    json.dump(predictions, f)

print('Predictions saved to albert_predictions.json')
