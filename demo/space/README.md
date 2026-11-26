---
title: BERT contre TF-IDF (SST-2)
emoji: 🎬
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
models: [IlyasBBB/bert-sst2-sentiment]
short_description: BERT fine-tuné contre TF-IDF, sur une phrase au choix
---

Démo du projet [bert-sst2-sentiment](https://github.com/IlyasBBB/bert-sst2-sentiment) : la même phrase passe par
BERT fine-tuné et par une baseline TF-IDF + régression logistique, avec des probabilités calibrées.
