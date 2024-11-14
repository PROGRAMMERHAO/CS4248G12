import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from tqdm import tqdm
import collections
import os
from torch.amp import autocast, GradScaler  # Updated imports for mixed precision

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# if using cuda, clear the cache first
if device == 'cuda':
    torch.cuda.empty_cache()
    print('Cache cleared')

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

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B', use_fast=True)
model = LlamaForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B')
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

# Mixed precision training
scaler = GradScaler(device='cuda')  # Updated GradScaler syntax
accumulation_steps = 4  # Increase accumulation to reduce batch memory

# Directory to save the best model
best_model_directory = '../models/llama_model'
if not os.path.exists(best_model_directory):
    os.makedirs(best_model_directory)

# Training loop with best model saving
model.train()
best_loss = float("inf")
for epoch in range(3):
    print(f'Epoch {epoch + 1}/3')
    loop = tqdm(train_loader, leave=True)
    for i, batch in enumerate(loop):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Mixed precision
        with autocast(device_type='cuda'):  # Updated autocast syntax
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accumulation_steps
        
        # Scale and backpropagate
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        current_loss = loss.item() * accumulation_steps
        loop.set_postfix(loss=current_loss)

        # Save the model if this is the lowest loss observed so far
        if current_loss < best_loss:
            best_loss = current_loss
            model.save_pretrained(best_model_directory)
            tokenizer.save_pretrained(best_model_directory)
            print(f"New best model saved with loss {best_loss:.4f} at {best_model_directory}")

# Save the final model
final_save_directory = '../models/llama_model'
if not os.path.exists(final_save_directory):
    os.makedirs(final_save_directory)
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
        input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=256).to(device)

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
if not os.path.exists('../eval'):
    os.makedirs('../eval')
with open('../eval/llama_predictions.json', 'w') as f:
    json.dump(predictions, f)

print('Predictions saved to llama_predictions.json')
