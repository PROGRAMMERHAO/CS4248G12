import os
import torch
import json
import collections
from tqdm import tqdm
from torch import nn
import os
import torch
import json
import collections
from tqdm import tqdm
from torch import nn
from transformers import (
    AlbertTokenizerFast,  # Use the Fast tokenizer
    AlbertModel,
    AlbertPreTrainedModel,
    AlbertConfig,
)

# Define the modified model with cross-attention
class AlbertWithCrossAttention(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Initialize the base ALBERT model
        self.albert = AlbertModel(config)
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=8)
        # Output layer for start and end logits
        self.output_layer = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Get ALBERT base model outputs
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        # Transpose for MultiheadAttention (requires shape: (seq_len, batch_size, embed_dim))
        sequence_output = sequence_output.transpose(0, 1)

        # Cross-attention over question and context embeddings
        cross_attn_output, _ = self.cross_attention(sequence_output, sequence_output, sequence_output)

        # Transpose back to (batch_size, seq_len, embed_dim)
        cross_attn_output = cross_attn_output.transpose(0, 1)

        # Pass through output layer for start and end logits
        logits = self.output_layer(cross_attn_output)

        # Separate start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # Shape: (batch_size, seq_len)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

# Instantiate the enhanced model with cross-attention
model_name = "ktrapeznikov/albert-xlarge-v2-squad-v2"
config = AlbertConfig.from_pretrained(model_name)
enhanced_model = AlbertWithCrossAttention(config)

# Load the pre-trained weights into the base ALBERT model
enhanced_model.albert = AlbertModel.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enhanced_model.to(device)
enhanced_model.eval()

# Load tokenizer
tokenizer = AlbertTokenizerFast.from_pretrained(model_name)

# Define directory to save the final model and tokenizer
final_save_directory = '../models/albert_model_with_cross_attention'
os.makedirs(final_save_directory, exist_ok=True)

# Save the enhanced model and tokenizer
enhanced_model.save_pretrained(final_save_directory)
tokenizer.save_pretrained(final_save_directory)
print(f'Model and tokenizer saved to {final_save_directory}')

# Load the SQuAD dev data
with open('../data/dev-v1.1.json', 'r') as f:
    squad_data = json.load(f)

# Preprocess the data
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

# Run predictions and save them
predictions = collections.OrderedDict()

# Custom prediction loop
with torch.no_grad():
    for item in tqdm(dev_data):
        question = item['question']
        context = item['context']
        qid = item['qid']

        # Tokenize the inputs
        inputs = tokenizer(
            question,
            context,
            max_length=384,
            truncation='only_second',
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(device)  # Shape: (num_spans, seq_len)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        offset_mapping = inputs['offset_mapping']

        # Get model predictions
        start_logits, end_logits = enhanced_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )  # Each is of shape (num_spans, seq_len)

        # For each span
        all_answers = []
        for i in range(len(input_ids)):
            start_logit = start_logits[i]
            end_logit = end_logits[i]
            offsets = offset_mapping[i]

            # Get the most probable start and end token positions
            start_indexes = torch.argsort(start_logit, descending=True)[:20].cpu().numpy()
            end_indexes = torch.argsort(end_logit, descending=True)[:20].cpu().numpy()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offsets) or end_index >= len(offsets):
                        continue
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if offsets[start_index][0] > offsets[end_index][1]:
                        continue
                    answer = context[offsets[start_index][0]:offsets[end_index][1]]
                    score = start_logit[start_index].item() + end_logit[end_index].item()
                    all_answers.append({'text': answer, 'score': score})

        # Choose the best answer
        if all_answers:
            best_answer = max(all_answers, key=lambda x: x['score'])['text']
        else:
            best_answer = ''

        predictions[qid] = best_answer

# Save predictions to a JSON file
with open(os.path.join(eval_directory, 'albert_refine_predictions.json'), 'w') as f:
    json.dump(predictions, f)

print('Predictions saved to albert_refine_predictions.json')
