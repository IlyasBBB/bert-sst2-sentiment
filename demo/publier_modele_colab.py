# Cellule à coller À LA FIN du notebook, dans Colab, après une exécution complète avec SAVE_MODEL = True.
# Elle publie sur le Hugging Face Hub :
#   - le BERT fine-tuné de cette exécution (poids + tokenizer, dossier results/bert_sst2_model) ;
#   - la baseline TF-IDF + régression logistique exportée en JSON (vocabulaire, idf, coefficients), lisible
#     quelle que soit la version de scikit-learn du Space ;
#   - les températures du temperature scaling et les métriques réellement mesurées pendant cette exécution.
# Le jeton Hugging Face (droit « write ») est demandé par notebook_login() : il n'est jamais écrit dans le notebook.

HF_REPO = "IlyasBBB/bert-sst2-sentiment"   # <nom d'utilisateur Hugging Face>/<nom du modèle>

import json
import numpy as np
from huggingface_hub import HfApi, notebook_login

notebook_login()
api = HfApi()
api.create_repo(HF_REPO, repo_type="model", exist_ok=True)

model_dir = OUT / "bert_sst2_model"
assert model_dir.exists(), "Relancer le notebook avec SAVE_MODEL = True"

# Baseline TF-IDF : tout ce qu'il faut pour refaire transform() + predict_proba() sans pickle
inv_vocab = {int(i): t for t, i in vec.vocabulary_.items()}
tfidf_export = {
    "ngram_range": list(vec.ngram_range), "sublinear_tf": bool(vec.sublinear_tf), "lowercase": bool(vec.lowercase),
    "token_pattern": vec.token_pattern, "norm": vec.norm,
    "vocabulary": [inv_vocab[i] for i in range(len(inv_vocab))],
    "idf": [round(float(x), 6) for x in vec.idf_],
    "coef": [round(float(x), 6) for x in clf.coef_[0]], "intercept": float(clf.intercept_[0]),
}
json.dump(tfidf_export, open(model_dir / "tfidf_baseline.json", "w"))

bert_m = json.load(open(OUT / "bert_metrics.json"))
tfidf_m = json.load(open(OUT / "tfidf_metrics.json"))
info = {
    "temperature_bert": float(CALIB["bert"]["temperature"]), "temperature_tfidf": float(CALIB["tfidf"]["temperature"]),
    "val_accuracy_bert": bert_m["val"]["accuracy"], "val_accuracy_tfidf": tfidf_m["val"]["accuracy"],
    "seed": bert_m["seed"], "max_len": MAX_LEN,
}
json.dump(info, open(model_dir / "demo_info.json", "w"), indent=2)

card = f"""---
language: en
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
datasets: [glue]
base_model: bert-base-uncased
tags: [sentiment-analysis, sst-2, bert]
---

# BERT fine-tuné sur SST-2 (analyse de sentiment)

`bert-base-uncased` fine-tuné sur SST-2 (GLUE), avec une boucle PyTorch écrite à la main.
Code, protocole et analyse : https://github.com/IlyasBBB/bert-sst2-sentiment

| Modèle | Accuracy, validation officielle SST-2 (872 phrases) |
|---|--:|
| BERT fine-tuné (ce modèle, graine {info['seed']}) | {info['val_accuracy_bert']:.2%} |
| TF-IDF + régression logistique (`tfidf_baseline.json`) | {info['val_accuracy_tfidf']:.2%} |

Étiquettes : 0 = négatif, 1 = positif. Les probabilités se calibrent en divisant les logits par
T = {info['temperature_bert']:.2f} (temperature scaling ajusté sur le jeu de développement).
Limites : critiques de films en anglais, phrases courtes ; l'ironie et les avis mitigés restent les erreurs les plus fréquentes.
"""
open(model_dir / "README.md", "w").write(card)

api.upload_folder(folder_path=str(model_dir), repo_id=HF_REPO, repo_type="model",
                  commit_message="BERT fine-tuné sur SST-2 et baseline TF-IDF")
print(f"Publié : https://huggingface.co/{HF_REPO}  (BERT {info['val_accuracy_bert']:.2%}, TF-IDF {info['val_accuracy_tfidf']:.2%})")
