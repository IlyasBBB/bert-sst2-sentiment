# ---
# complement: H
# titre: Export des erreurs de BERT pour la lecture manuelle
# ancre: a9728992
# ordre: 4
# actif: oui
# ---

# %% [markdown]
# ### 9.8 · Erreurs de BERT à relire
# Le cours demande dix phrases mal classées, regroupées par cause : négation, ironie, avis mitigé, vocabulaire rare, étiquette discutable. La cellule suivante rassemble toutes les erreurs de BERT avec ce qui aide à les lire (probabilités des deux modèles, marqueurs de négation et d'opposition, mots hors vocabulaire de la baseline) et les enregistre dans `results/erreurs_bert_a_annoter.csv`. Les plus confiantes viennent en premier, car ce sont celles où le modèle se trompe sans hésiter. La lecture elle-même est présentée en section 9.9.

# %%
if HAS_BERT:
    ERR_BERT = (T[~T.ok_bert]
                .assign(confiance=lambda d: np.maximum(d.p_bert, 1 - d.p_bert), baseline_juste=lambda d: d.ok_tfidf,
                        oov_mots=lambda d: d.oov_mots.str.join(" "))
                .sort_values("confiance", ascending=False)
                [["idx", "sentence", "label", "p_bert", "p_tfidf", "confiance", "baseline_juste", "negation", "contrast", "n_words", "oov_mots"]])
    ERR_BERT.to_csv(OUT / "erreurs_bert_a_annoter.csv", index=False)
    print(f"{len(ERR_BERT)} erreurs de BERT enregistrées, dont {int((ERR_BERT.confiance >= CONFIDENT).sum())} "
          f"avec une confiance d'au moins {fr(CONFIDENT)}.")
    display(ERR_BERT.head(12).assign(label=lambda d: d.label.map({0: "négatif", 1: "positif"}))
            .style.format({"p_bert": "{:.3f}", "p_tfidf": "{:.3f}", "confiance": "{:.3f}"}).hide(axis="index"))
