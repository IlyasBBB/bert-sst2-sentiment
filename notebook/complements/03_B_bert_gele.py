# ---
# complement: B
# titre: BERT gelé contre fine-tuning complet (bonus du cours)
# ancre: 1479b737
# ordre: 1
# actif: oui
# ---

# %% [markdown]
# ### 7.1 · Bonus du cours : BERT gelé contre fine-tuning complet
# Le cours propose de comparer le fine-tuning complet, où les 110 M de poids sont ajustés, à une version où BERT reste gelé et où seule une couche de classification est apprise. On procède en deux temps. Le BERT pré-entraîné, sans aucune adaptation, encode chaque phrase ; une régression logistique est ensuite entraînée sur ces vecteurs, ce qui se rapproche de n'entraîner que la tête de classification. Deux représentations sont testées : le vecteur de sortie du jeton `[CLS]`, point de départ de la tête de classification de BERT (qui lui applique encore une couche dense pré-entraînée, le *pooler*), et la moyenne des vecteurs de tous les jetons. Le paramètre `C` est choisi sur `dev` et la validation officielle n'est évaluée qu'une fois par variante.

# %%
from sklearn.preprocessing import StandardScaler

def load_best_bert(tag="main"):
    """Recharge le meilleur checkpoint du fine-tuning (choisi sur dev), en mode évaluation."""
    with silence_hf():
        m = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / f"best_{tag}.pt", map_location=device))
    return m.eval()

FROZEN_RESULTS = {}
if BERT_AVAILABLE and RUN_FROZEN:
    from transformers import AutoModel
    t_cellule = time.time()
    set_seed()
    with silence_hf():
        encoder = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()  # poids pré-entraînés, aucun fine-tuning

    @torch.no_grad()
    def encode(texts, bs=EVAL_BATCH):
        """Vecteur [CLS] et moyenne des jetons non masqués, en sortie de la dernière couche."""
        cls_, mean_ = [], []
        for i in range(0, len(texts), bs):
            b = tokenizer(texts[i:i + bs], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
            with autocast_ctx():
                h = encoder(**b).last_hidden_state.float()
            mask = b["attention_mask"].unsqueeze(-1).float()
            cls_.append(h[:, 0].cpu().numpy())
            mean_.append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
        return np.concatenate(cls_), np.concatenate(mean_)

    fit_df = train_fit if FROZEN_MAX_TRAIN is None else train_fit.sample(min(FROZEN_MAX_TRAIN, len(train_fit)), random_state=SEED)
    t0 = time.time()
    EMB = {name: encode(df.sentence.tolist()) for name, df in [("train", fit_df), ("dev", dev_df), ("val", val_df)]}
    embed_min = (time.time() - t0) / 60
    for k, rep in enumerate(["cls", "mean"]):
        key = f"frozen_{rep}"
        sc = StandardScaler().fit(EMB["train"][k])
        Xtr_, Xdev_, Xval_ = (sc.transform(EMB[s][k]) for s in ("train", "dev", "val"))
        grid_rows = []
        for C_ in FROZEN_C_GRID:                                                # sélection de C sur dev uniquement
            lr_ = LogisticRegression(C=C_, max_iter=5000).fit(Xtr_, fit_df.label)
            grid_rows.append({"C": C_, "dev_acc": accuracy_score(dev_df.label, lr_.predict(Xdev_))})
        best_C = float(pd.DataFrame(grid_rows).sort_values(["dev_acc", "C"], ascending=[False, True]).iloc[0].C)
        lr_ = LogisticRegression(C=best_C, max_iter=5000).fit(Xtr_, fit_df.label)
        p_dev_, p_val_ = lr_.predict_proba(Xdev_)[:, 1], lr_.predict_proba(Xval_)[:, 1]
        FROZEN_RESULTS[key] = {"dev": compute_metrics(dev_df.label, p_dev_), "val": evaluate_final(key, p_val_)}
        save_preds(key, "dev", dev_df, p_dev_)
        json.dump({"model": f"{MODEL_NAME} gelé ({'[CLS]' if rep == 'cls' else 'moyenne des jetons'}) + LogReg",
                   "representation": rep, "C": best_C, "n_train": len(fit_df), "embed_minutes": embed_min,
                   "grid": grid_rows, **FROZEN_RESULTS[key]}, open(OUT / f"{key}_metrics.json", "w"), indent=2)
    del encoder, EMB, Xtr_, Xdev_, Xval_
    free_gpu()
    total_min = (time.time() - t_cellule) / 60                                 # encodage + toutes les régressions
    for key in FROZEN_RESULTS:
        f_ = OUT / f"{key}_metrics.json"; d_ = json.load(open(f_)); d_["total_minutes"] = total_min
        json.dump(d_, open(f_, "w"), indent=2)

    lignes = [("TF-IDF + LogReg", base_metrics), ("BERT gelé, [CLS] + LogReg", FROZEN_RESULTS["frozen_cls"]),
              ("BERT gelé, moyenne + LogReg", FROZEN_RESULTS["frozen_mean"]), ("BERT fine-tuné", bert_metrics)]
    cols_acc = ["accuracy dev", "accuracy validation", "F1 validation", "ROC-AUC validation"]
    display(pd.DataFrame([{"modèle": n, "accuracy dev": r["dev"]["accuracy"], "accuracy validation": r["val"]["accuracy"],
                           "F1 validation": r["val"]["f1"], "ROC-AUC validation": r["val"]["roc_auc"]} for n, r in lignes])
            .style.format({c: "{:.4f}" for c in cols_acc}).hide(axis="index"))
    print(f"Encodage des trois jeux par le BERT gelé : {embed_min:.1f} min sur {device.upper()} ; "
          f"cellule complète, régressions comprises : {total_min:.1f} min.")
else:
    print("BERT gelé non évalué (RUN_FROZEN = False ou modèle indisponible).")
