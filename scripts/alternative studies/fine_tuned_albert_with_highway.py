# qa_system.py

import os
import json
import torch
import collections
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel, BertForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the data
def read_squad(path):
    with open(path, 'r') as f:
        squad = json.load(f)
    data = []
    for article in squad['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                qid = qa['id']
                answers = qa['answers']
                if len(answers) == 0:
                    continue  # Skip unanswerable questions
                answer_text = answers[0]['text']
                answer_start = answers[0]['answer_start']
                data.append({
                    'context': context,
                    'question': question,
                    'qid': qid,
                    'answer_text': answer_text,
                    'answer_start': answer_start
                })
    return data

train_data = read_squad('../data/train-v1.1.json')
dev_data = read_squad('../data/dev-v1.1.json')

# Create a custom dataset
class SquadDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=384, doc_stride=128):
        self.examples = []
        for item in tqdm(data, desc="Processing Data"):
            inputs = tokenizer(
                item['question'],
                item['context'],
                max_length=max_length,
                truncation='only_second',
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding='max_length'
            )
            offset_mapping = inputs.pop('offset_mapping')
            sample_mapping = inputs.pop('overflow_to_sample_mapping')
            answers = item['answer_text']
            start_char = item['answer_start']
            end_char = start_char + len(answers)
            for i in range(len(inputs['input_ids'])):
                input_ids = inputs['input_ids'][i]
                attention_mask = inputs['attention_mask'][i]
                token_type_ids = inputs['token_type_ids'][i]
                offsets = offset_mapping[i]
                sample_idx = sample_mapping[i]
                cls_index = input_ids.index(tokenizer.cls_token_id)
                sequence_ids = inputs.sequence_ids(i)

                # Find the start and end of the context in the input_ids
                context_start = 0
                while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
                    context_start += 1
                context_end = len(sequence_ids) - 1
                while context_end >= 0 and sequence_ids[context_end] != 1:
                    context_end -= 1
                if context_start > context_end:
                    continue  # Skip if context not found

                if not (start_char >= offsets[context_start][0] and end_char <= offsets[context_end][1]):
                    start_positions = cls_index
                    end_positions = cls_index
                else:
                    start_positions = end_positions = None
                    for idx, (offset_start, offset_end) in enumerate(offsets):
                        if offset_start == offset_end:
                            continue
                        if offset_start <= start_char and offset_end >= start_char:
                            start_positions = idx
                        if offset_start <= end_char and offset_end >= end_char:
                            end_positions = idx
                    if start_positions is None:
                        start_positions = cls_index
                    if end_positions is None:
                        end_positions = cls_index
                self.examples.append({
                    'input_ids': torch.tensor(input_ids),
                    'attention_mask': torch.tensor(attention_mask),
                    'token_type_ids': torch.tensor(token_type_ids),
                    'start_positions': torch.tensor(start_positions),
                    'end_positions': torch.tensor(end_positions)
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

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
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x
        return x

# Define BertLinear_highway class
class BertLinear_highway(nn.Module):
    def __init__(self, model_name):
        super(BertLinear_highway, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        input_dim = self.bert.config.hidden_size  # Get hidden size from config
        self.enc = HighwayEncoder(3, input_dim)
        self.qa_outputs = nn.Linear(input_dim, 2)
        
    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.enc(sequence_output)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits)
        if start_positions is not None and end_positions is not None:
            # Compute loss
            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits

# Initialize tbertokenizer and model
model_name = "-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertLinear_highway(model_name=model_name)

# Move model to device
model.to(device)

# Create datasets and dataloaders
train_dataset = SquadDataset(train_data, tokenizer)
dev_dataset = SquadDataset(dev_data, tokenizer)

# Adjust batch size due to larger model size
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=8)

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=3e-5)
total_steps = len(train_loader) * 2  # 2 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

epoch_num = 2
best_loss = float("inf")  # Initialize best loss

# Directory to save the best model
best_model_directory = '../models/fine_tuned_model_bert'
if not os.path.exists(best_model_directory):
    os.makedirs(best_model_directory)

# Training loop
model.train()
for epoch in range(epoch_num):
    print(f'Epoch {epoch + 1}/{epoch_num}')
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions
        )
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_loss = loss.item()
        loop.set_postfix(loss=current_loss)
        
        # Save the model if this is the lowest loss observed so far
        if current_loss < best_loss:
            best_loss = current_loss
            model_save_path = os.path.join(best_model_directory, 'model.pt')
            torch.save(model.state_dict(), model_save_path)
            tokenizer.save_pretrained(best_model_directory)
            print(f"New best model saved with loss {best_loss:.4f} at {best_model_directory}")

# Save the final model
final_save_directory = '../models/fine_tuned_model_bert'
if not os.path.exists(final_save_directory):
    os.makedirs(final_save_directory)

model_save_path = os.path.join(final_save_directory, 'model.pt')
torch.save(model.state_dict(), model_save_path)
tokenizer.save_pretrained(final_save_directory)

print(f'Model and tokenizer saved to {final_save_directory}')

# Prediction code and saving predictions to JSON
model.eval()
predictions = collections.OrderedDict()
with torch.no_grad():
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

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        start_logits, end_logits = outputs
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

# Ensure the eval directory exists
eval_directory = '../eval'
if not os.path.exists(eval_directory):
    os.makedirs(eval_directory)

# Save predictions to a JSON file
with open(os.path.join(eval_directory, 'fine_tuned_predictions_bert.json'), 'w') as f:
    json.dump(predictions, f)

print('Predictions saved to fine_tuned_predictions_bert.json')
