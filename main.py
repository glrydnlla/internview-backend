from fastapi import FastAPI, Request
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

class QAPair(BaseModel):
    question: str
    answer: str

class QARequest(BaseModel):
    qa_list: List[QAPair]
    
class TAPair(BaseModel):
    title: str
    article: str

class TARequest(BaseModel):
    article_list: List[TAPair]

def qa_to_text(qa_list):
    return "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in qa_list])

def article_to_text(article_list):
    return "\n".join([f"{item['title']}\n{item['article']}\n" for item in article_list])

@app.post("/generate-article")
def generate_article(data: QARequest):
    text = qa_to_text([qa.dict() for qa in data.qa_list])
    prompt = "Buat ringkasan menggunakan bahasa indonesia dari wawancara ini dalam bentuk artikel paragraf, dengan tujuan membantu orang lain yang juga akan mengikuti wawancara yang sama. Tolong bandingkan juga jawaban peserta yang lulus dan tidak, apa yang membuat mereka lulus atau tidak lulus? sertakan jawabannya dalam artikel.  jika menemukan kalimat satir atau kata-kata tidak pantas, jangan dimasukkan ke dalam artikel, pastikan yang dimasukkan ke artikel hanya hal-hal yang berguna untuk membantu orang lain yang ingin melamar di pekerjaan yang sama. jangan sebutkan nama orang (pelamar) sama sekali, gunakan kata \"para pelamar\" untuk menggantikan nama orang (pelamar), nama perusahaan, kampus, atau institusi boleh disebutkan. \nPastikan ringkasannya dalam bentuk artikel, jangan diformat bold atau italic, tidak pakai point, tidak pakai \'*\'. Berikan Judul yang menarik dan unik yang merepresentasikan artikel tersebut.\n" + text

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
def generate_article(data: TARequest):
    text = article_to_text([ta.dict() for ta in data.article_list])
    prompt = "Buat rangkuman tentang artikel berikut, tidak usah diberi judul. poin pentingnya adalah apa kunci sukses dan hal yang harus dihindari supaya lulus interview yang mereka jalani secara singkat padat jelas (tidak lebih dari 1000 kata)\n\n" + text
    completion = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    summary = completion.choices[0].message.content
    summary = summary.replace("*", "")
    return {"summary": summary}