import random
import re
import string
import json
from flask import Flask, render_template, request

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# === Load konten medis dalam bahasa Indonesia ===
with open('medical research.txt', 'r', errors='ignore') as f:
    content = f.read()

# === Pisahkan kalimat secara sederhana ===
sentence_tokens = re.split(r'[.!?]', content)
sentence_tokens = [sent.strip() for sent in sentence_tokens if sent.strip()]

# === Preprocessing sederhana ===
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # hanya huruf
    text = re.sub(r'\s+', ' ', text)         # normalisasi spasi
    return text.strip()

processed_sentences = [preprocess_text(sent) for sent in sentence_tokens]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_sentences)

# === Daftar sapaan ===
greeting_inputs = ('hai', 'halo', 'apa kabar', 'selamat pagi', 'selamat malam')
greeting_responses = [
    "Halo! Saya di sini untuk membantu masalah kesehatan Anda.",
    "Hai! Silakan ceritakan keluhan atau gejala Anda.",
    "Selamat datang! Apa yang ingin Anda konsultasikan hari ini?",
    "Halo! Yuk, bicara soal kesehatan Anda."
]

def greet(user_input):
    for word in user_input.split():
        if word.lower() in greeting_inputs:
            return random.choice(greeting_responses)
    return None

# === Generator respons TF-IDF ===
def generate_response(user_input):
    processed_input = preprocess_text(user_input)
    processed_sentences.append(processed_input)
    tfidf = vectorizer.transform(processed_sentences)
    cosine_sim = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = cosine_sim.argsort()[0][-1]
    flat = cosine_sim.flatten()
    flat.sort()
    score = flat[-2]
    processed_sentences.pop()

    if score == 0:
        return "Maaf, saya tidak memahami itu. Bisa dijelaskan lebih lanjut?"
    else:
        return sentence_tokens[idx]

# === Pola aturan berbasis keyword ===
try:
    with open('rules.json') as f:
        rules = json.load(f)
except FileNotFoundError:
    rules = []

def get_rule_based_response(user_input):
    for rule in rules:
        if re.search(rule["pattern"], user_input, re.IGNORECASE):
            return rule["response"]
    return None

# === Flask Routes ===
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def respond():
    user_input = request.form["user_input"].lower()

    if user_input == "selesai" or user_input == "terima kasih":
        return "Terima kasih telah menggunakan layanan chatbot kami. Semoga lekas sembuh!"

    greet_msg = greet(user_input)
    if greet_msg:
        return greet_msg

    rule_msg = get_rule_based_response(user_input)
    if rule_msg:
        return rule_msg

    return generate_response(user_input)

if __name__ == "__main__":
    app.run(debug=True)
