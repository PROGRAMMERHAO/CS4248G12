# llama_qa_system.py

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM, AdamW, get_linear_schedule_with_warmup
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

# Custom Dataset for LLaMA
class SquadDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.examples = []
        for item in tqdm(data, desc="Processing Data"):
            question = item['question']
            context = item['context']
            answer = item['answer_text']
            input_text = f"Question: {question}\nContext: {context}\nAnswer:"
            full_text = input_text + " " + answer

            encoding = tokenizer(
                full_text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
                add_special_tokens=True
            )

            input_ids = encoding['input_ids'][0]
            attention_mask = encoding['attention_mask'][0]

            # Determine the position where the answer starts
            input_text_encoding = tokenizer(
                input_text,
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
            input_len = input_text_encoding['input_ids'].size(1)

            labels = input_ids.clone()
            labels[:input_len] = -100  # Set labels for input tokens to -100

            self.examples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Initialize tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')

# Set the pad_token_id
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

model.to(device)

# DataLoaders
train_dataset = SquadDataset(train_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # Adjust batch size based on your GPU memory
dev_dataset = SquadDataset(dev_data, tokenizer)
dev_loader = DataLoader(dev_dataset, batch_size=1)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=3e-5)
total_steps = len(train_loader) * 3  # for 3 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

epoch_num = 3
best_loss = float("inf")  # Initialize best loss

# Directory to save the best model
best_model_directory = '../models/llama_model'
if not os.path.exists(best_model_directory):
    os.makedirs(best_model_directory)

# Training loop with best model saving
model.train()
for epoch in range(epoch_num):
    print(f'Epoch {epoch + 1}/{epoch_num}')
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_loss = loss.item()
        loop.set_postfix(loss=current_loss)

        # Save the model if this is the lowest loss observed so far
        if current_loss < best_loss:
            best_loss = current_loss
            model.save_pretrained(best_model_directory)
            tokenizer.save_pretrained(best_model_directory)
            print(f"New best model saved with loss {best_loss:.4f} at {best_model_directory}")

# Define a directory to save the final model
final_save_directory = '../models/llama_model'
if not os.path.exists(final_save_directory):
    os.makedirs(final_save_directory)

# Save the final model
model.save_pretrained(final_save_directory)
tokenizer.save_pretrained(final_save_directory)

print(f'Model and tokenizer saved to {final_save_directory}')

# Prediction code and saving predictions to JSON
model.eval()
predictions = collections.OrderedDict()
with torch.no_grad():
    for item in tqdm(dev_data):
        question = item['question']
        context = item['context']
        qid = item['qid']
        input_text = f"Question: {question}\nContext: {context}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=1024).to(device)

        # Generate answer
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            num_return_sequences=1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Extract the answer by removing the input_text from the generated_text
        if generated_text.startswith(input_text):
            answer = generated_text[len(input_text):].strip()
        else:
            answer = generated_text

        predictions[qid] = answer

# Save predictions to a JSON file
with open('../eval/llama_predictions.json', 'w') as f:
    json.dump(predictions, f)

print('Predictions saved to llama_predictions.json')
