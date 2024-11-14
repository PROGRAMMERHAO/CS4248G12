# qa_system.py

import os
import json
import torch
import collections
from tqdm import tqdm
from transformers import AlbertTokenizerFast, AlbertModel
import torch.nn as nn
import torch.nn.functional as F

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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

# Define HighwayEncoder class
class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network."""
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            g = torch.sigmoid(gate(x))
            t = torch.relu(transform(x))
            x = g * t + (1 - g) * x
        return x

# Define AlbertLinear_highway class
class AlbertLinear_highway(nn.Module):
    def __init__(self, model_name):
        super(AlbertLinear_highway, self).__init__()
        # Load the pre-trained ALBERT model
        self.albert = AlbertModel.from_pretrained(model_name)
        input_dim = self.albert.config.hidden_size  # Get hidden size from config
        self.enc = HighwayEncoder(3, input_dim)
        self.qa_outputs = nn.Linear(input_dim, 2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
        sequence_output = self.enc(sequence_output)

        logits = self.qa_outputs(sequence_output)  # Shape: (batch_size, seq_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)  # Shape: (batch_size, seq_len, 1)
        start_logits = start_logits.squeeze(-1)  # Shape: (batch_size, seq_len)
        end_logits = end_logits.squeeze(-1)      # Shape: (batch_size, seq_len)

        return start_logits, end_logits

# Initialize tokenizer
model_name = "ktrapeznikov/albert-xlarge-v2-squad-v2"
tokenizer = AlbertTokenizerFast.from_pretrained(model_name)

# Initialize the custom model
model = AlbertLinear_highway(model_name)

# Load the fine-tuned model weights from model.pt
model_save_path = '../models/fine_tuned_model/model.pt'
if os.path.exists(model_save_path):
    state_dict = torch.load(model_save_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded fine-tuned model weights from {model_save_path}")
else:
    print(f"Model weights not found at {model_save_path}")
    # Optionally, you can load the pre-trained weights if model.pt is not found
    # and then fine-tune the model as needed.

# Move model to device
model.to(device)
model.eval()

# Ensure the eval directory exists
eval_directory = '../eval'
os.makedirs(eval_directory, exist_ok=True)

# Prediction code and saving predictions to JSON
predictions = collections.OrderedDict()

for item in tqdm(dev_data):
    inputs = tokenizer(
        item['question'],
        item['context'],
        max_length=384,
        truncation='only_second',
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)
    offset_mapping = inputs['offset_mapping']
    sample_mapping = inputs['overflow_to_sample_mapping']

    # Get model predictions
    with torch.no_grad():
        start_logits, end_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    # For each input
    for i in range(len(input_ids)):
        offsets = offset_mapping[i]
        start_logit = start_logits[i]
        end_logit = end_logits[i]
        start_indexes = torch.argsort(start_logit, descending=True)[:20]
        end_indexes = torch.argsort(end_logit, descending=True)[:20]
        context = item['context']
        valid_answers = []
        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index >= len(offsets) or end_index >= len(offsets):
                    continue
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                if offsets[start_index][0] > offsets[end_index][1]:
                    continue
                answer = context[offsets[start_index][0]:offsets[end_index][1]]
                valid_answers.append({
                    'text': answer,
                    'score': start_logit[start_index].item() + end_logit[end_index].item()
                })
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x['score'], reverse=True)[0]['text']
        else:
            best_answer = ''
        predictions[item['qid']] = best_answer

# Save predictions to a JSON file
with open(os.path.join(eval_directory, 'albert_predictions.json'), 'w') as f:
    json.dump(predictions, f)

print('Predictions saved to albert_predictions.json')
