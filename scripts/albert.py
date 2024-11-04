# qa_system.py

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AlbertTokenizerFast,
    AlbertForQuestionAnswering,
    AlbertPreTrainedModel,
    AlbertConfig,
    AdamW,
    get_linear_schedule_with_warmup
)
import torch.nn as nn
from tqdm import tqdm
import collections
import os

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
                    continue  # Skip unanswerable questions (shouldn't be any in SQuAD 1.1)
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

# Use SQuAD 1.1 dataset
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
                cls_index = input_ids.index(tokenizer.cls_token_id)
                sequence_ids = inputs.sequence_ids(i)
                context_start = sequence_ids.index(1)
                context_end = len(sequence_ids) - sequence_ids[::-1].index(1)

                # Find the start and end positions within the tokenized context
                start_positions = end_positions = None
                for idx, (offset_start, offset_end) in enumerate(offsets):
                    if offset_start <= start_char < offset_end:
                        start_positions = idx
                    if offset_start < end_char <= offset_end:
                        end_positions = idx

                if start_positions is None or end_positions is None:
                    # If the answer is not fully inside the context, use cls_index
                    start_positions = cls_index
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

# Define a custom model with additional layers
class CustomAlbertForQuestionAnswering(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # Should be 2 (start and end positions)
        self.albert = AlbertForQuestionAnswering.from_pretrained('albert-base-v2').albert
        self.bilstm = nn.LSTM(config.hidden_size, config.hidden_size, bidirectional=True, batch_first=True)
        self.self_attention = nn.MultiheadAttention(config.hidden_size * 2, num_heads=8, batch_first=True)
        self.highway = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
            nn.Sigmoid()
        )
        self.qa_outputs = nn.Linear(config.hidden_size * 2, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        start_positions=None,
        end_positions=None
    ):
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]  # (batch_size, seq_length, hidden_size)

        # Pass through BiLSTM
        lstm_output, _ = self.bilstm(sequence_output)
        lstm_output = self.dropout(lstm_output)

        # Self-Attention
        attn_output, _ = self.self_attention(
            lstm_output, lstm_output, lstm_output,
            key_padding_mask=~attention_mask.bool()
        )
        attn_output = self.dropout(attn_output)

        # Highway Network
        highway_gate = self.highway(attn_output)
        highway_output = highway_gate * attn_output + (1 - highway_gate) * lstm_output
        highway_output = self.dropout(highway_output)

        # Output layer
        logits = self.qa_outputs(highway_output)  # (batch_size, seq_length, num_labels)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (batch_size, seq_length)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits) + outputs[2:]  # Add hidden states and attentions if they are here

        if start_positions is not None and end_positions is not None:
            # Compute loss
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

# Initialize tokenizer and model
tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')
config = AlbertConfig.from_pretrained('albert-base-v2')
config.num_labels = 2  # For start and end positions
model = CustomAlbertForQuestionAnswering(config)
model.to(device)

# Create datasets and dataloaders
train_dataset = SquadDataset(train_data, tokenizer)
dev_dataset = SquadDataset(dev_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=8)

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=3e-5)
total_steps = len(train_loader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

epoch_num = 3
best_loss = float("inf")  # Initialize best loss

# Directory to save the best model
best_model_directory = '../models/fine_tuned_model'
if not os.path.exists(best_model_directory):
    os.makedirs(best_model_directory)

# Training loop
for epoch in range(epoch_num):
    print(f'Epoch {epoch + 1}/{epoch_num}')
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        outputs = model(
            input_ids,
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

    # Save the model at the end of each epoch
    model.save_pretrained(best_model_directory)
    tokenizer.save_pretrained(best_model_directory)
    print(f"Model saved at {best_model_directory}")

# Define a directory to save the final model and tokenizer
final_save_directory = '../models/fine_tuned_model'
if not os.path.exists(final_save_directory):
    os.makedirs(final_save_directory)

# Save the final model
model.save_pretrained(final_save_directory)
tokenizer.save_pretrained(final_save_directory)

print(f'Model and tokenizer saved to {final_save_directory}')

# Ensemble Prediction Code
# Assuming you have trained and saved multiple models (e.g., fine_tuned_model1, fine_tuned_model2, fine_tuned_model3)
model_paths = [
    '../models/fine_tuned_model1',
    '../models/fine_tuned_model2',
    '../models/fine_tuned_model3'
]

models = []
for model_path in model_paths:
    model = CustomAlbertForQuestionAnswering.from_pretrained(model_path)
    model.to(device)
    model.eval()
    models.append(model)

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

        # Collect logits from each model
        start_logits_list = []
        end_logits_list = []
        for model in models:
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            start_logits_list.append(outputs[0])  # start_logits
            end_logits_list.append(outputs[1])    # end_logits

        # Aggregate logits (mean)
        start_logits = torch.stack(start_logits_list).mean(dim=0)
        end_logits = torch.stack(end_logits_list).mean(dim=0)

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
                        'score': start_logit[start_index] + end_logit[end_index]
                    })
            if len(valid_answers) > 0:
                best_answer = sorted(valid_answers, key=lambda x: x['score'], reverse=True)[0]['text']
            else:
                best_answer = ''
            predictions[item['qid']] = best_answer

# Save predictions to a JSON file
with open('../eval/fine_tuned_predictions.json', 'w') as f:
    json.dump(predictions, f)

print('Predictions saved to fine_tuned_predictions.json')
