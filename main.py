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

class QAPair(BaseModel):
    question: str
    answer: str

class QARequest(BaseModel):
    qa_list: List[QAPair]
    prompt: str
    
class TAPair(BaseModel):
    title: str
    article: str

class SummaryRequest(BaseModel):
    article_list: List[TAPair]
    prompt: str
    

def qa_to_text(qa_list):
    return "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in qa_list])

def article_to_text(article_list):
    return "\n".join([f"{item['title']}\n{item['article']}\n" for item in article_list])

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
def generate_article(data: SummaryRequest):
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