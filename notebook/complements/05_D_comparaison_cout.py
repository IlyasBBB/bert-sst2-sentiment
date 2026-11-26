# ---
# complement: D
# titre: Comparaison élargie et coût d'inférence
# ancre: ede0fe3d
# ordre: 1
# actif: oui
# ---

# %% [markdown]
# ### 8.1 · Comparaison élargie et coût d'inférence
# Le premier tableau réunit tous les modèles évalués sur la validation officielle, avec l'intervalle de Wilson à 95 %. Les tests appariés (McNemar exact et IC bootstrap de l'écart) comparent ensuite BERT fine-tuné à la meilleure variante gelée, choisie sur `dev`, puis cette variante à TF-IDF. On mesure enfin le temps de prédiction des 872 phrases et la taille des deux modèles principaux, puisque le choix d'un modèle ne se fait pas sur l'accuracy seule.

# %%
def pair_test(key_a, key_b):
    """McNemar exact et IC bootstrap apparié de l'écart d'accuracy (b − a) sur la validation officielle."""
    A = pd.read_csv(OUT / f"{key_a}_val_preds.csv"); B_ = pd.read_csv(OUT / f"{key_b}_val_preds.csv")
    assert (A.idx.values == B_.idx.values).all(), "plan apparié : mêmes phrases dans le même ordre"
    ok_a, ok_b = (A.pred == A.label).values, (B_.pred == B_.label).values
    n10, n01 = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    p = binomtest(n01, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    lo_, hi_ = paired_boot(ok_a, ok_b)
    return {"a": key_a, "b": key_b, "acc_a": float(ok_a.mean()), "acc_b": float(ok_b.mean()),
            "ecart": float(ok_b.mean() - ok_a.mean()), "ic_bas": float(lo_), "ic_haut": float(hi_),
            "a_seul_juste": n10, "b_seul_juste": n01, "p_mcnemar": float(p)}

MODEL_LABELS = {"tfidf": "TF-IDF + LogReg", "frozen_cls": "BERT gelé, [CLS] + LogReg",
                "frozen_mean": "BERT gelé, moyenne + LogReg", "distilbert": "DistilBERT fine-tuné", "bert": "BERT fine-tuné"}
dispo = [k for k in MODEL_LABELS if (OUT / f"{k}_val_preds.csv").exists()]
tab = []
for k in dispo:
    v = pd.read_csv(OUT / f"{k}_val_preds.csv"); ok_ = (v.pred == v.label).values
    lo_w_, hi_w_ = wilson(int(ok_.sum()), len(ok_))
    tab.append({"modèle": MODEL_LABELS[k], "accuracy": ok_.mean(), "IC95 Wilson": f"[{lo_w_:.3f} ; {hi_w_:.3f}]",
                "F1 (positif)": f1_score(v.label, v.pred), "ROC-AUC": roc_auc_score(v.label, v.proba)})
display(pd.DataFrame(tab).style.format({"accuracy": "{:.4f}", "F1 (positif)": "{:.4f}", "ROC-AUC": "{:.4f}"}).hide(axis="index"))

PAIRS = []
gelees = [k for k in ("frozen_cls", "frozen_mean") if k in dispo]
if gelees:
    BEST_FROZEN = max(gelees, key=lambda k: json.load(open(OUT / f"{k}_metrics.json"))["dev"]["accuracy"])  # choix sur dev
    if "bert" in dispo:
        PAIRS.append(pair_test(BEST_FROZEN, "bert"))
    PAIRS.append(pair_test("tfidf", BEST_FROZEN))
if "distilbert" in dispo and "bert" in dispo:
    PAIRS.append(pair_test("distilbert", "bert"))
if PAIRS:
    json.dump(PAIRS, open(OUT / "comparaisons_appariees.json", "w"), indent=2)
    display(pd.DataFrame(PAIRS).assign(a=lambda d: d.a.map(MODEL_LABELS), b=lambda d: d.b.map(MODEL_LABELS))
            .style.format({"acc_a": "{:.4f}", "acc_b": "{:.4f}", "ecart": "{:+.4f}", "ic_bas": "{:+.4f}",
                           "ic_haut": "{:+.4f}", "p_mcnemar": "{:.2e}"}).hide(axis="index"))

# %%
# ---------- Coût d'inférence sur les 872 phrases de validation ----------
COUT = {"device": device}
durees_ = []
for _ in range(6):                                                       # 1 préchauffage + médiane de 5 mesures
    t0 = time.perf_counter(); clf.predict_proba(vec.transform(val_df.sentence)); durees_.append(time.perf_counter() - t0)
COUT["tfidf_s"] = float(np.median(durees_[1:]))
COUT["tfidf_params"] = int(clf.coef_.size + clf.intercept_.size)
if BERT_AVAILABLE:
    bert_ft = load_best_bert("main")
    # Ces passages sur la validation servent au chronométrage et au contrôle du rechargement : aucune décision n'en dépend.
    evaluate(bert_ft, val_df)                                            # préchauffage (chargement des noyaux GPU)
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter(); _, p_check = evaluate(bert_ft, val_df)
    if device == "cuda": torch.cuda.synchronize()
    COUT["bert_s"] = time.perf_counter() - t0
    COUT["bert_params"] = int(sum(p.numel() for p in bert_ft.parameters()))
    COUT["bert_mo_float32"] = COUT["bert_params"] * 4 / 1e6
    COUT["ratio_temps"] = COUT["bert_s"] / COUT["tfidf_s"]
    COUT["ecart_max_rechargement"] = float(np.abs(pd.read_csv(OUT / "bert_val_preds.csv").proba.values - p_check).max())
    assert COUT["ecart_max_rechargement"] < 1e-2, "le checkpoint rechargé ne reproduit pas les prédictions enregistrées"
    del bert_ft
    free_gpu()
json.dump(COUT, open(OUT / "cout_metrics.json", "w"), indent=2)
display(pd.DataFrame([{"modèle": "TF-IDF + LogReg", "paramètres": COUT["tfidf_params"], "prédiction des 872 phrases (s)": COUT["tfidf_s"], "matériel": "CPU"}]
                     + ([{"modèle": "BERT fine-tuné", "paramètres": COUT["bert_params"], "prédiction des 872 phrases (s)": COUT["bert_s"], "matériel": device.upper()}] if "bert_s" in COUT else []))
        .style.format({"paramètres": "{:,}", "prédiction des 872 phrases (s)": "{:.3f}"}).hide(axis="index"))
if "bert_s" in COUT:
    print(f"Contrôle : le checkpoint rechargé reproduit les probabilités enregistrées (écart max {COUT['ecart_max_rechargement']:.1e}).")
