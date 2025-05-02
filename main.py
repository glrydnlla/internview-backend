from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from openai import OpenAI
import sqlite3

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

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
    
class TAPair(BaseModel):
    title: str
    article: str

class SummaryRequest(BaseModel):
    article_list: List[TAPair]
    prompt: str
    
class SKPair(BaseModel):
    rank: int
    softskill: str

class SummaryRequest(BaseModel):
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
    prompt = data.prompt + text

    completion = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    article = completion.choices[0].message.content
    article = article.replace("*", "")
    
    sentences = sent_tokenize(article)
    title = sentences[0]
    
    print(title)
    article = article.replace(title, "").lstrip()

    return {"title": title, "article": article}


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
def find_job(data: SummaryRequest):
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

@app.post("/process-form")
def process_form_data(data: QARequest):
    text = qa_list_to_text([qa.dict() for qa in data.qa_list])
    prompt = data.prompt + text
    completion = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    result = completion.choices[0].message.content
    result = result.replace("*", "")
    
    sentences = sent_tokenize(result)
    title = sentences[0]
    
    print(title)
    result = result.replace(title, "").lstrip()

    return {"title": title, "result": result}