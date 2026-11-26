# ---
# complement: E
# titre: Hypothèse H4 (mots hors vocabulaire) et écarts entre sous-groupes
# ancre: a9728992
# ordre: 1
# actif: oui
# ---

# %% [markdown]
# ### 9.5 · Mise à l'épreuve des hypothèses H1, H2 et H4
# La section 3 a compté les mots distincts de la validation qui n'apparaissent jamais dans le train. TF-IDF ne peut rien faire de ces mots, alors que BERT les découpe en unités WordPiece rencontrées pendant son pré-entraînement. Si H4 est juste, l'avantage de BERT doit être plus grand sur les phrases qui contiennent au moins un mot absent du vocabulaire de la baseline. Ce vocabulaire est plus étroit que celui du train : TF-IDF ne garde que les mots vus au moins deux fois dans `train_fit` et ignore les mots d'une seule lettre. Le tableau 9.1 donnait un écart par sous-groupe ; pour conclure qu'un écart est *plus grand* dans un groupe qu'en dehors, il faut un intervalle de confiance sur la différence des deux écarts. La seconde cellule le calcule par bootstrap pour chaque marqueur : négation et opposition (H1), plus de 30 mots (H2), mot hors vocabulaire (H4).

# %%
# ---------- Mots absents du vocabulaire de la baseline (unigrammes de TF-IDF) ----------
analyseur = vec.build_analyzer()
def mots_hors_vocab(s):
    return sorted({w for w in analyseur(s) if " " not in w and w not in vec.vocabulary_})
T["oov_mots"] = T.sentence.apply(mots_hors_vocab)
T["oov"] = T.oov_mots.str.len() > 0

lignes_oov = []
for nom, masque in {"toutes les phrases": pd.Series(True, index=T.index),
                    "sans mot hors vocabulaire": ~T.oov, "avec mot hors vocabulaire": T.oov}.items():
    r = {"sous-groupe": nom, "n": int(masque.sum())}
    for m in MODELS:
        k_ = int(T.loc[masque, f"ok_{m}"].sum()); lo_, hi_ = wilson(k_, int(masque.sum()))
        r[f"acc {LAB[m]}"] = k_ / max(int(masque.sum()), 1); r[f"IC95 {LAB[m]}"] = f"[{lo_:.3f} ; {hi_:.3f}]"
    if HAS_BERT and masque.sum() > 1:
        l_, h_ = paired_boot(T.loc[masque, "ok_tfidf"].values, T.loc[masque, "ok_bert"].values)
        r["écart BERT − base"] = r["acc BERT"] - r["acc TF-IDF + LogReg"]; r["IC95 écart (bootstrap)"] = f"[{l_:+.3f} ; {h_:+.3f}]"
    lignes_oov.append(r)
G_OOV = pd.DataFrame(lignes_oov).set_index("sous-groupe")
G_OOV.to_csv(OUT / "sous_groupes_oov.csv")
display(G_OOV.style.format({**{c: "{:.3f}" for c in G_OOV.columns if c.startswith("acc")}, "écart BERT − base": "{:+.3f}"}))
OOV_STATS = {"n_avec": int(T.oov.sum()), "pct_avec": float(T.oov.mean() * 100),
             "mots_distincts": int(len(set().union(*T.oov_mots))), "vocabulaire_tfidf_unigrammes": int(sum(" " not in w for w in vec.vocabulary_))}
for m in MODELS:
    err_ = ~T[f"ok_{m}"]
    odds_, p_ = fisher_exact([[int((err_ & T.oov).sum()), int((err_ & ~T.oov).sum())],
                              [int((~err_ & T.oov).sum()), int((~err_ & ~T.oov).sum())]])
    OOV_STATS[f"{m}_acc_avec"] = float(T.loc[T.oov, f"ok_{m}"].mean()); OOV_STATS[f"{m}_acc_sans"] = float(T.loc[~T.oov, f"ok_{m}"].mean())
    OOV_STATS[f"{m}_fisher_odds"] = float(odds_); OOV_STATS[f"{m}_fisher_p"] = float(p_)
json.dump(OOV_STATS, open(OUT / "oov_metrics.json", "w"), indent=2)

# %%
# ---------- Différence des écarts BERT − baseline : dans le groupe contre hors du groupe ----------
def ecart_des_ecarts(masque, n_boot=N_BOOT, seed=SEED):
    """Écart (BERT − base) dans le groupe moins écart hors du groupe, avec IC bootstrap à 95 %."""
    rng_ = np.random.default_rng(seed)
    a_in, b_in = T.loc[masque, "ok_tfidf"].values, T.loc[masque, "ok_bert"].values
    a_out, b_out = T.loc[~masque, "ok_tfidf"].values, T.loc[~masque, "ok_bert"].values
    i_in = rng_.integers(0, len(a_in), (n_boot, len(a_in))); i_out = rng_.integers(0, len(a_out), (n_boot, len(a_out)))
    d = (b_in[i_in].mean(1) - a_in[i_in].mean(1)) - (b_out[i_out].mean(1) - a_out[i_out].mean(1))
    obs = (b_in.mean() - a_in.mean()) - (b_out.mean() - a_out.mean())
    return float(obs), *map(float, np.percentile(d, [2.5, 97.5]))

if HAS_BERT:
    marqueurs = {"négation (H1)": T.negation, "opposition (H1)": T.contrast,
                 "plus de 30 mots (H2)": T.n_words > 30, "mot hors vocabulaire (H4)": T.oov}
    lignes_dd = []
    for nom, masque in marqueurs.items():
        obs, lo_, hi_ = ecart_des_ecarts(masque.values)
        lignes_dd.append({"marqueur": nom, "n avec": int(masque.sum()),
                          "écart avec": T.loc[masque, "ok_bert"].mean() - T.loc[masque, "ok_tfidf"].mean(),
                          "écart sans": T.loc[~masque, "ok_bert"].mean() - T.loc[~masque, "ok_tfidf"].mean(),
                          "différence": obs, "IC95 bas": lo_, "IC95 haut": hi_,
                          "conclusion": "avantage de BERT plus grand dans le groupe" if lo_ > 0 else
                                        ("avantage plus petit dans le groupe" if hi_ < 0 else "pas de différence établie")})
    DD = pd.DataFrame(lignes_dd)
    DD.to_csv(OUT / "ecart_des_ecarts.csv", index=False)
    display(DD.style.format({"écart avec": "{:+.3f}", "écart sans": "{:+.3f}", "différence": "{:+.3f}",
                             "IC95 bas": "{:+.3f}", "IC95 haut": "{:+.3f}"}).hide(axis="index"))
    display(Markdown("*Lecture : une différence dont l'IC à 95 % exclut 0 indique que l'avantage de BERT dépend du marqueur. "
                     "Quatre comparaisons exploratoires, sans correction pour tests multiples : un résultat isolé proche du seuil reste fragile.*"))

# Table d'analyse complète de la validation, réutilisée en 9.8 et pour le rapport
T.assign(oov_mots=T.oov_mots.str.join(" ")).to_csv(OUT / "analyse_validation.csv", index=False)
G.to_csv(OUT / "sous_groupes.csv")
assoc.to_csv(OUT / "association_marqueurs.csv", index=False)
prof.to_csv(OUT / "profil_erreurs.csv", index=False)
