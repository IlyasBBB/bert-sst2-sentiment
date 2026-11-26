# ---
# complement: C
# titre: Option DistilBERT (désactivée par défaut)
# ancre: 1479b737
# ordre: 2
# actif: oui
# ---

# %% [markdown]
# ### 7.2 · Option : DistilBERT, la version allégée citée dans le cours
# Le cours présente `distilbert-base-uncased` comme l'option des machines sans GPU : un modèle distillé à partir de BERT, plus petit et plus rapide, presque aussi bon selon ses auteurs. Avec `RUN_DISTILBERT = True`, il est fine-tuné avec exactement le même protocole (mêmes données, mêmes hyper-paramètres, époque choisie sur `dev`) pour mesurer ce compromis sur SST-2.

# %%
# ecrase-volontairement: MODEL_NAME tokenizer   (restaurés dans le bloc finally)
DISTIL_METRICS = None
if BERT_AVAILABLE and RUN_DISTILBERT:
    _sauvegarde = (MODEL_NAME, tokenizer)                 # fine_tune et make_loader lisent ces deux globales
    try:
        MODEL_NAME = DISTIL_MODEL_NAME
        with silence_hf():
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model_d, hist_d, sec_d = fine_tune(SEED, tag="distil")
        _, pdv_d = evaluate(model_d, dev_df)
        _, pvl_d = evaluate(model_d, val_df)
        DISTIL_METRICS = {"dev": compute_metrics(dev_df.label, pdv_d), "val": evaluate_final("distilbert", pvl_d)}
        save_preds("distilbert", "dev", dev_df, pdv_d)
        json.dump({"model": MODEL_NAME, "epochs": EPOCHS, "n_params": int(sum(p.numel() for p in model_d.parameters())),
                   "train_minutes": sec_d / 60, "best_epoch": int(pd.DataFrame(hist_d).dev_acc.idxmax() + 1), **DISTIL_METRICS},
                  open(OUT / "distilbert_metrics.json", "w"), indent=2)
        display(pd.DataFrame(DISTIL_METRICS).T.rename(index={"dev": "dev (sélection)", "val": "validation officielle"}).round(4))
        del model_d
        free_gpu()
    finally:
        MODEL_NAME, tokenizer = _sauvegarde
else:
    print("DistilBERT non évalué (RUN_DISTILBERT = False).")
