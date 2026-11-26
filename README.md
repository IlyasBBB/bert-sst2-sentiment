# Fine-tuning de BERT pour la classification de sentiment (SST-2)

Travaux pratiques de NLP (module 7, Transformer et BERT) réalisés à l'École Centrale Casablanca, option S2D, par HABBANI Mohammed, BEN ALLA Ilyas et ETTAHIRI Mohammed-Yassine (novembre 2026).

Nous comparons trois façons de classer le sentiment des phrases de SST-2 (GLUE) : une baseline TF-IDF + régression logistique, BERT gelé suivi d'une régression logistique, et `bert-base-uncased` fine-tuné. Les choix (hyper-paramètres, époque, variante) se font sur un jeu de développement interne, et la validation officielle n'est évaluée qu'une fois par modèle. Le test GLUE n'a pas d'étiquettes publiques et n'est pas utilisé.

## Résultats

Accuracy sur les 872 phrases de la validation officielle, exécution complète sur Google Colab (GPU Tesla T4) :

| Modèle | Accuracy |
|---|--:|
| TF-IDF + régression logistique | 81,88 % |
| BERT gelé (moyenne des jetons) + régression logistique | 85,32 % |
| BERT fine-tuné | 92,66 % |

- L'écart entre BERT fine-tuné et la baseline vaut +10,78 points, avec un IC bootstrap apparié à 95 % de [+8,14 ; +13,42] et un test de McNemar à p = 3,4 × 10⁻¹⁵.
- Avec trois graines (42, 43, 44), l'accuracy du fine-tuning va de 92,43 % à 93,12 %.
- Le fine-tuning complet ajoute +7,3 points à la version gelée, avec un IC de [+5,2 ; +9,4].
- Seul l'avantage de BERT sur les phrases de plus de 30 mots est établi. Pour la négation, l'opposition et les mots hors vocabulaire, l'intervalle contient 0.

Le rapport détaille aussi la calibration des probabilités, des paires minimales avec et sans négation et dix erreurs de BERT classées par cause.

## Contenu

| Chemin | Contenu |
|---|---|
| `notebook/SST2_BERT_vs_TFIDF_NLP_Projet.ipynb` | notebook complet, exécuté de bout en bout, avec ses sorties |
| `notebook/complements/` | code des cellules ajoutées à la première version du notebook (sections 6.1, 7.1, 7.2, 8.1, 9.5 à 9.9, 10.3, 10.4) |
| `rapport/Rapport_TP_NLP_BERT_SST2_NLP_Projet.pdf` | rapport de 16 pages |
| `rapport/rapport.md` | texte du rapport en Markdown, lisible sur GitHub |
| `resultats/` | métriques (JSON), prédictions par phrase (CSV), figures et annotation des erreurs |

## Relancer le notebook

Le plus simple est Google Colab : importer le notebook, choisir un GPU T4 (Exécution > Modifier le type d'exécution), puis tout exécuter. La première cellule installe les dépendances. Compter environ 50 minutes, dont la moitié pour les deux graines supplémentaires ; mettre `RUN_EXTRA_SEEDS = False` dans la cellule de paramètres des compléments pour aller plus vite. La dernière cellule télécharge `results_sst2.zip`, qui contient les mêmes fichiers que `resultats/`.

En local, avec Python 3.10 ou plus :

```
pip install -r requirements.txt
```

Sans GPU, le fine-tuning complet de BERT prend plusieurs heures.

## Références principales

- Devlin, J., Chang, M.-W., Lee, K. et Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT.
- Socher, R. et al. (2013). Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. EMNLP.
- Wang, A. et al. (2019). GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. ICLR.

La liste complète est à la fin du rapport.
