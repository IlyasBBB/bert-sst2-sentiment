# ---
# complement: F
# titre: Phrases personnelles et paires minimales (H1)
# ancre: a9728992
# ordre: 2
# actif: oui
# ---

# %% [markdown]
# ### 9.6 · Phrases personnelles et paires minimales
# Le cours demande de tester le modèle sur ses propres phrases. On le fait de deux façons. D'abord douze phrases écrites pour l'occasion, avec des négations, des oppositions, de l'ironie et des mots rares. Ensuite des paires minimales construites à partir de gabarits. La même phrase apparaît avec et sans négation (« the movie is good », puis « the movie is not good »). Des phrases à deux propositions reliées par *but* complètent le jeu : l'avis retenu y est celui de la seconde proposition. Ces phrases isolent la négation et l'opposition, ce que les mots-clés de la section 9.1 ne permettaient pas. Elles testent H1 de façon contrôlée, sur un anglais plus simple que celui des critiques réelles. Convention d'étiquetage : une phrase niée reçoit l'étiquette inverse de l'affirmation (« not bad » compte comme positif, comme le suggère le poids très positif que la baseline donne à ce bigramme en section 5 ; « not brilliant » compte comme négatif). Ce choix se discute pour certains adjectifs et pèse sur l'accuracy des gabarits de négation. Le taux d'inversion, lui, n'en dépend pas : il dit seulement si la prédiction change quand on ajoute la négation.

# %%
def predict_both(texts):
    """Probabilité « positif » selon la baseline et selon BERT fine-tuné (meilleur checkpoint)."""
    out = pd.DataFrame({"phrase": texts, "p_tfidf": clf.predict_proba(vec.transform(texts))[:, 1]})
    if BERT_AVAILABLE:
        m_ = load_best_bert("main"); probs = []
        with torch.no_grad():
            for i in range(0, len(texts), EVAL_BATCH):
                b = tokenizer(texts[i:i + EVAL_BATCH], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
                with autocast_ctx():
                    probs.append(torch.softmax(m_(**b).logits.float(), -1)[:, 1].cpu().numpy())
        out["p_bert"] = np.concatenate(probs)
        del m_
        free_gpu()
    return out

# ---- Phrases personnelles (étiquette attendue fixée par nous) ----
PERSO = [("this movie is not bad at all .", 1),
         ("i expected a masterpiece , and it is not even watchable .", 0),
         ("the actors are great , but the script is a disaster .", 0),
         ("the script is a disaster , but the actors are great .", 1),
         ("oh great , another two hours of my life i will never get back .", 0),
         ("a film so bad it becomes hilarious , in the best way .", 1),
         ("it is hard to imagine a less inspired sequel .", 0),
         ("not a single scene feels false .", 1),
         ("the cinematography is breathtaking ; the dialogue , sadly , is not .", 0),
         ("i laughed , i cried , i bought the soundtrack .", 1),
         ("a perfunctory , by-the-numbers thriller .", 0),
         ("an unpretentious , sweet-natured delight .", 1)]
PP = predict_both([s for s, _ in PERSO]).assign(attendu=[lab for _, lab in PERSO])
for m in [x for x in ("tfidf", "bert") if f"p_{x}" in PP]:
    PP[f"{m} juste"] = np.where((PP[f"p_{m}"] >= .5).astype(int) == PP.attendu, "oui", "non")
PP["attendu"] = PP.attendu.map({0: "négatif", 1: "positif"})
PP.to_csv(OUT / "phrases_personnelles.csv", index=False)
display(PP.style.format({c: "{:.2f}" for c in ("p_tfidf", "p_bert") if c in PP}).hide(axis="index"))

# %%
# ---- Paires minimales construites par gabarits ----
SUJETS = ["the movie", "this film", "the story", "the acting", "the ending"]
POSITIFS = ["good", "great", "funny", "moving", "brilliant", "enjoyable"]
NEGATIFS = ["bad", "boring", "dull", "awful", "predictable", "tedious"]
lignes_mp = []
for s in SUJETS:
    for adj, lab in [(w, 1) for w in POSITIFS] + [(w, 0) for w in NEGATIFS]:
        paire = f"{s}|{adj}"
        lignes_mp += [{"gabarit": "affirmation", "phrase": f"{s} is {adj} .", "label": lab, "paire": paire},
                      {"gabarit": "négation « not »", "phrase": f"{s} is not {adj} .", "label": 1 - lab, "paire": paire},
                      {"gabarit": "négation « n't »", "phrase": f"{s} is n't {adj} .", "label": 1 - lab, "paire": paire}]
for s1, s2 in [("the acting", "the story"), ("the music", "the plot"), ("the cast", "the script")]:
    for adj_pos in POSITIFS[:4]:                  # noms de boucle choisis pour ne pas écraser « b » (section 8)
        for adj_neg in NEGATIFS[:4]:
            lignes_mp += [{"gabarit": "opposition « + but − »", "phrase": f"{s1} is {adj_pos} , but {s2} is {adj_neg} .", "label": 0, "paire": None},
                          {"gabarit": "opposition « − but + »", "phrase": f"{s1} is {adj_neg} , but {s2} is {adj_pos} .", "label": 1, "paire": None}]
MP = pd.DataFrame(lignes_mp)
MP = pd.concat([MP, predict_both(MP.phrase.tolist()).drop(columns="phrase")], axis=1)
MODELES_MP = [m for m in ("tfidf", "bert") if f"p_{m}" in MP]
for m in MODELES_MP:
    MP[f"pred_{m}"] = (MP[f"p_{m}"] >= .5).astype(int); MP[f"ok_{m}"] = MP[f"pred_{m}"] == MP.label
MP.to_csv(OUT / "paires_minimales.csv", index=False)

resume_mp, MP_STATS = [], {}
for gab, d in MP.groupby("gabarit", sort=False):
    r = {"gabarit": gab, "n": len(d)}
    for m in MODELES_MP:
        lo_, hi_ = wilson(int(d[f"ok_{m}"].sum()), len(d))
        r[f"acc {LAB[m]}"] = d[f"ok_{m}"].mean(); r[f"IC95 {LAB[m]}"] = f"[{lo_:.2f} ; {hi_:.2f}]"
        MP_STATS[f"{m}|{gab}"] = float(d[f"ok_{m}"].mean())
    resume_mp.append(r)
# Taux d'inversion : la prédiction change-t-elle quand on ajoute la négation ?
for m in MODELES_MP:
    for gab in ["négation « not »", "négation « n't »"]:
        aff = MP[MP.gabarit == "affirmation"].set_index("paire")[f"pred_{m}"]
        neg = MP[MP.gabarit == gab].set_index("paire")[f"pred_{m}"]
        MP_STATS[f"{m}|inversion {gab}"] = float((aff != neg.loc[aff.index]).mean())
json.dump({"n_phrases": len(MP), **MP_STATS}, open(OUT / "paires_minimales_metrics.json", "w"), indent=2)
display(pd.DataFrame(resume_mp).style.format({c: "{:.2f}" for c in pd.DataFrame(resume_mp).columns if c.startswith("acc")}).hide(axis="index"))
display(pd.DataFrame([{"modèle": LAB[m], **{f"inversion {g.split(' ', 1)[1]}": MP_STATS[f"{m}|inversion {g}"]
                                            for g in ["négation « not »", "négation « n't »"]}} for m in MODELES_MP])
        .style.format({c: "{:.0%}" for c in ["inversion « not »", "inversion « n't »"]}).hide(axis="index"))
display(Markdown("*Taux d'inversion : part des paires où ajouter la négation change la prédiction. Un modèle qui traite "
                 "correctement la négation inverse sa prédiction dans 100 % des paires.*"))
