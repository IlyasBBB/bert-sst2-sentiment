"""Démo Gradio : BERT fine-tuné contre TF-IDF + régression logistique sur une phrase en anglais.

Le modèle et la baseline viennent du Hugging Face Hub (dépôt MODEL_REPO), publiés depuis le notebook
(demo/publier_modele_colab.py). Les probabilités affichées sont calibrées par temperature scaling.
"""

import json
import math
import re
from collections import Counter

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_REPO = "IlyasBBB/bert-sst2-sentiment"
GITHUB = "https://github.com/IlyasBBB/bert-sst2-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO).eval()
info = json.load(open(hf_hub_download(MODEL_REPO, "demo_info.json")))
tf = json.load(open(hf_hub_download(MODEL_REPO, "tfidf_baseline.json")))

VOCAB = {term: i for i, term in enumerate(tf["vocabulary"])}
IDF = np.array(tf["idf"])
COEF = np.array(tf["coef"])
TOKEN_RE = re.compile(tf["token_pattern"])
NGRAMS = range(tf["ngram_range"][0], tf["ngram_range"][1] + 1)


def tfidf_proba(text: str, temperature: float) -> tuple[float, list[str]]:
    """Same steps as scikit-learn's TfidfVectorizer (sublinear tf, idf, l2) then the logistic regression."""
    words = TOKEN_RE.findall(text.lower() if tf["lowercase"] else text)
    grams = [" ".join(words[i:i + n]) for n in NGRAMS for i in range(len(words) - n + 1)]
    counts = Counter(g for g in grams if g in VOCAB)
    if not counts:
        return 1 / (1 + math.exp(-tf["intercept"] / temperature)), []
    idx = np.array([VOCAB[g] for g in counts])
    vals = np.array([(1 + math.log(c)) if tf["sublinear_tf"] else c for c in counts.values()]) * IDF[idx]
    vals /= np.linalg.norm(vals) or 1.0
    logit = float(vals @ COEF[idx]) + tf["intercept"]
    weights = sorted(zip(counts, vals * COEF[idx]), key=lambda kv: -abs(kv[1]))[:5]
    return 1 / (1 + math.exp(-logit / temperature)), [f"{g} ({w:+.2f})" for g, w in weights]


@torch.inference_mode()
def bert_proba(text: str, temperature: float) -> float:
    enc = tokenizer(text, truncation=True, max_length=info["max_len"], return_tensors="pt")
    logits = model(**enc).logits[0] / temperature
    return float(torch.softmax(logits, dim=-1)[1])


def analyse(text: str):
    text = (text or "").strip()
    if not text:
        return {}, {}, ""
    p_bert = bert_proba(text, info["temperature_bert"])
    p_tfidf, top = tfidf_proba(text, info["temperature_tfidf"])
    detail = ("**N-grammes qui pèsent le plus pour TF-IDF :** " + ", ".join(top)) if top else \
        "TF-IDF ne connaît aucun mot de cette phrase : il ne peut que renvoyer sa probabilité par défaut."
    return ({"positif": p_bert, "négatif": 1 - p_bert}, {"positif": p_tfidf, "négatif": 1 - p_tfidf}, detail)


EXAMPLES = [
    "the movie is not good .",
    "the movie is n't bad at all .",
    "a gorgeous , witty , seductive movie .",
    "i wanted to like it , but the plot falls apart in the second half .",
    "it 's the kind of film that makes you want to leave before the end .",
    "oh great , another two hours of my life i will never get back .",
]

with gr.Blocks(title="BERT contre TF-IDF (SST-2)") as demo:
    gr.Markdown(
        f"# BERT contre TF-IDF : analyse de sentiment\n"
        f"Tapez une critique de film **en anglais**. Les deux modèles ont été entraînés sur SST-2 ; "
        f"accuracy sur la validation officielle : BERT {info['val_accuracy_bert']:.1%}, "
        f"TF-IDF {info['val_accuracy_tfidf']:.1%}. Probabilités calibrées (temperature scaling). "
        f"Méthode et analyse des erreurs : [{GITHUB}]({GITHUB}).")
    text = gr.Textbox(label="Phrase", lines=2, placeholder="the acting is great but the story is boring")
    button = gr.Button("Analyser", variant="primary")
    with gr.Row():
        out_bert = gr.Label(label="BERT fine-tuné")
        out_tfidf = gr.Label(label="TF-IDF + régression logistique")
    out_detail = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=text)
    gr.Markdown("Limites : phrases courtes en anglais, critiques de films ; l'ironie et les avis mitigés "
                "restent les erreurs les plus fréquentes de BERT.")
    button.click(analyse, text, [out_bert, out_tfidf, out_detail])
    text.submit(analyse, text, [out_bert, out_tfidf, out_detail])

if __name__ == "__main__":
    demo.launch()
