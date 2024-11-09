import json
import torch
import re
import string
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
def remove_articles(text):
    regex = re.compile(r'^\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, '', text).lstrip()

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

dev_data = read_squad('../data/dev-v1.1.json')

output_parser_qa = StrOutputParser()
output_parser_compression = StrOutputParser()
output_parser_check = StrOutputParser()

prompt_qa = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in extactive question answering."),
    ("user", "{input}")
])

prompt_compression = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in extactive question answering."),
    ("user", "{input}")
])

prompt_check = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in understanding texts and question answering with no prior backgraound knowledge."),
    ("user", "{input}")
])

llm_qa = Ollama(model="llama3")
llm_compression = Ollama(model="llama3")
llm_check = Ollama(model="llama3")
chain_qa = prompt_qa | llm_qa | output_parser_qa
chain_compression = prompt_compression | llm_compression | output_parser_compression
chain_check = prompt_check | llm_check | output_parser_check
results = {}
max_attempts = 10
for testcase in dev_data:
    passage = testcase['context']
    question = testcase['question']
    inputs_qa = f"Given the following passage, answer the following question with the most relevant text.\
        You should only respond with the exactly same text from part of the passage. NO EXPLANATION NEEDED. DO NOT REPEAT the question or contain unnecessary information.\
        Passage: {passage}\
        Question: {question}\
        "
    qid = testcase['qid']
    print(qid)
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        pred = chain_qa.invoke({"input": inputs_qa})
        print(pred)
        inputs_compression = f"Given the question and answer below, extract and compress the answer to include only the key information needed to directly answer the question. \
            DO NOT modify the text or add details; only select the most relevant portions of the text. \
            First, ensure the response is MEANINGFUL and directly addresses the question (avoid trivial statements like 'A is A'). \
            Second, ensure the response is as concise as possible (but not overly concise). \
            Respond with ONLY the compressed answer in the exactly same text (no modifications, extra words or explanations). No need to be full sentence.\
            Question: {question} \
            Answer: {pred}"

        pred = chain_compression.invoke({"input": inputs_compression})
        print(pred)
        if pred.lower() in passage.lower() or remove_punc(pred.lower()) in passage.lower() or remove_articles(remove_punc(pred.lower())) in passage.lower():
            inputs_check = f"Given the relevant context, question, and compressed answer below, verify if the answer logically addresses the question. \
                You should respond with 'yes' only if the answer provides a logical and meaningful response to the question, even if it may be factually incorrect. \
                Reject situations where the answer is merely a restatement of the question. \
                Respond with ONLY 'yes' or 'no'. No explanation needed. \
                Question: {question} \
                Answer: {pred} \
                Context: {passage}"

            isValid = chain_check.invoke({"input": inputs_check})
            print(isValid)
            if "yes" in isValid.lower():
                break
    
    print(pred)
    print(testcase['answer_text'])
    results[qid] = pred
    with open("results.txt", "a") as f:
        f.write(f"{qid}:{pred}\n")
json.dump(results, open("results.json", "w"), indent=2)