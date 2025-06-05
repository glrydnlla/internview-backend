from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from openai import OpenAI
import sqlite3

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import json

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QA(BaseModel):
    question: str
    answer: str

class QARequest(BaseModel):
    qa_list: List[QA]
    prompt: str
    keys: str
    tags: str
    
class TAPair(BaseModel):
    title: str
    article: str

class SummaryRequest(BaseModel):
    article_list: List[TAPair]
    prompt: str
    
class SKPair(BaseModel):
    rank: int
    softskill: str

class SoftskillSummaryRequest(BaseModel):
    softskill_list: List[SKPair]
    prompt: str
    

def qa_to_text(qa_list):
    return "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in qa_list])

def article_to_text(article_list):
    return "\n".join([f"{item['title']}\n{item['article']}\n" for item in article_list])

def rank_to_text(softskill_list):
    return "\n".join([f"{item['rank']}. {item['softskill']}\n" for item in softskill_list])

def qa_list_to_text(qa_list):
    text = ""
    for item in qa_list:
        if item['question_type'] == "draggable-list":
            ans = rank_to_text(item['answer'])
        else:
            ans = item['answer']
        text += f"Q: {item['question']}\nA: {ans}\n"
    return text

@app.post("/generate-article")
def generate_article(data: QARequest):
    text = qa_to_text([qa.dict() for qa in data.qa_list])
    prompt = data.prompt + text + f"\nRespond with a valid JSON object with keys: {data.keys}." + f"\nIf there is a key named tags or tag, choose the tags from this list: {data.tags}"
    # print(prompt)

    completion = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    res = completion.choices[0].message.content
    print(res)
    res = res.replace("*", "")
    # print(res)
    json_str = res[res.find('{') : res.rfind('}') + 1]
    # print(json_str)
    dict_result = json.loads(json_str)
    # print(dict_result)
    
    # sentences = sent_tokenize(article)
    # title = sentences[0]
    
    # print(title)
    # article = article.replace(title, "").lstrip()

    return dict_result


@app.post("/summarize")
def summarize(data: SummaryRequest):
    text = article_to_text([ta.dict() for ta in data.article_list])
    prompt = data.prompt + text
    completion = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    summary = completion.choices[0].message.content
    summary = summary.replace("*", "")
    return {"summary": summary}

@app.post("/find-job")
def find_job(data: SoftskillSummaryRequest):
    text = rank_to_text([ta.dict() for sf in data.softskill_list])
    prompt = data.prompt + text
    completion = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    job_summary = completion.choices[0].message.content
    job_summary = job_summary.replace("*", "")
    return {"job_summary": job_summary}

# @app.post("/process-form")
# def process_form_data(data: QARequest):
#     text = qa_list_to_text([qa.dict() for qa in data.qa_list])
#     prompt = data.prompt + text
#     completion = client.chat.completions.create(
#         model="deepseek/deepseek-r1:free",
#         messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )
#     result = completion.choices[0].message.content
#     result = result.replace("*", "")
    
#     sentences = sent_tokenize(result)
#     title = sentences[0]
    
#     print(title)
#     result = result.replace(title, "").lstrip()

#     return {"title": title, "result": result}