import json
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForQuestionAnswering, AlbertTokenizerFast, AlbertForQuestionAnswering
from tqdm import tqdm
import collections

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load tokenizers and models
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
albert_tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')
albert_model = AlbertForQuestionAnswering.from_pretrained('albert-base-v2').to(device)

# Load the dataset
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
                    continue
                data.append({
                    'context': context,
                    'question': question,
                    'qid': qid,
                })
    return data

dev_data = read_squad('../data/dev-v2.0.json')

# Prediction function for each model
def get_model_predictions(model, tokenizer, data):
    predictions = collections.OrderedDict()
    model.eval()
    with torch.no_grad():
        for item in tqdm(data):
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
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            offset_mapping = inputs['offset_mapping']
            sample_mapping = inputs['overflow_to_sample_mapping']

            # Get best prediction per item
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
                        score = start_logit[start_index] + end_logit[end_index]
                        valid_answers.append({
                            'text': answer,
                            'score': score.item()
                        })
                if valid_answers:
                    best_answer = sorted(valid_answers, key=lambda x: x['score'], reverse=True)[0]
                    predictions[item['qid']] = best_answer
                else:
                    predictions[item['qid']] = {'text': '', 'score': 0}
    return predictions

# Get predictions from both models
bert_predictions = get_model_predictions(bert_model, bert_tokenizer, dev_data)
albert_predictions = get_model_predictions(albert_model, albert_tokenizer, dev_data)

# Voting mechanism
final_predictions = {}
for qid in bert_predictions.keys():
    bert_answer = bert_predictions[qid]
    albert_answer = albert_predictions[qid]
    # Voting by higher score
    if bert_answer['score'] >= albert_answer['score']:
        final_predictions[qid] = bert_answer['text']
    else:
        final_predictions[qid] = albert_answer['text']
    # Or you could use both models' agreement:
    # final_predictions[qid] = bert_answer['text'] if bert_answer['text'] == albert_answer['text'] else (longer or higher-scoring answer)

# Save final predictions to JSON
with open('../eval/voting_albert_bert_predictions.json', 'w') as f:
    json.dump(final_predictions, f)

print('Voting-based predictions saved to voting_albert_bert_predictions.json')
