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
to_skip = 0
skipped = 0
for testcase in dev_data:
    if skipped < to_skip:
        skipped += 1
        continue
    passage = testcase['context']
    question = testcase['question']
    inputs_qa = f"Given the following passage, extract the answer of the following question from the passage.\
        You should only respond with the EXACTLY SAME text from part of the passage (no slice within a single word). Make sure your answer can directly address the question.\
        NO EXPLANATION NEEDED. DO NOT REPEAT the question or contain unnecessary information.\
        Passage: {passage}\
        Question: {question}\
        "
    qid = testcase['qid']
    print(qid)
    attempt = 0
    ans = None
    to_check = False
    while attempt < max_attempts:
        attempt += 1
        first_pred = chain_qa.invoke({"input": inputs_qa})
        #print(pred)
        inputs_compression = f"Given the question and answer below, extract and compress the answer to include only the key information needed to directly answer the question. \
            DO NOT modify the text or add details; only select the most relevant portions of the text. \
            First, ensure the response is MEANINGFUL and directly addresses the question (avoid trivial statements like 'AAA is AAA' or 'AAA is short for AAA'). \
            Second, ensure the response is as concise as possible, following the below points:\
                - might be only few words\
                - Directly answer the question. DO NOT repeat the question statement. No need to complete the sentence.\
                - No slice within a singe word allowed. You need to keep the full word. \
                - You shall only keep conjunctions, preposition and other linking words only if specificly asked (e.g., respond with '2016' instead of 'since 2016' if not specificly asked) \
                - You should only keep acticles if it is in the raw text, DO NOT add it by yourself.\
                - If the question asks for specific information such as a number, date, name, or location, respond with only that information, without any additional descriptors or context (e.g., respond with '2' instead of '2 apples').\
                - If the question asks for team name, brand name or other names, DO NOT slice within the name.\
            Respond with ONLY the compressed answer in the exactly same text. No modifications, extra words or explanations needed.\
            Question: {question} \
            Answer: {first_pred}"

        pred = chain_compression.invoke({"input": inputs_compression})
        print(pred)
        if pred.lower() in question.lower() or remove_punc(pred.lower()) in question.lower(): # Based on the assumption that the answer is not in the question
            to_check = not to_check
            if to_check:
                continue
        if pred.lower() in passage.lower() or remove_punc(pred.lower()) in passage.lower():
            '''
            inputs_check = f"Given the relevant context, question, and compressed answer below, verify if the answer logically addresses the question. \
                You should respond with 'yes' only if the answer provides a logical, meaningful and correct response to the question based on the context, even if the context may be factually incorrect. \
                Reject situations where the answer is merely a restatement of the question. \
                Respond with ONLY 'yes' or 'no'. No explanation needed. \
                Question: {question} \
                Answer: {pred} \
                Context: {passage}"

            isValid = chain_check.invoke({"input": inputs_check})
            print(isValid)
            '''
            ans = pred
            break
    if ans is None:
        ans = pred
    
    print(pred)
    print(testcase['answer_text'])
    results[qid] = pred
    with open("results.txt", "a", encoding='utf8') as f:
        f.write(f"{qid}:{pred}\n")
json.dump(results, open("results.json", "w"), indent=2)