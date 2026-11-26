# ---
# complement: A
# titre: Tokenisation WordPiece, [CLS], [SEP] et rembourrage
# ancre: 59afc000
# ordre: 1
# actif: oui
# ---

# %% [markdown]
# ### 6.1 · Ce que voit BERT : WordPiece, `[CLS]`, `[SEP]` et rembourrage
# BERT ne lit pas des mots mais des unités WordPiece, tirées d'un vocabulaire de 30 522 entrées pour `bert-base-uncased`. Un mot fréquent reste entier. Un mot rare est découpé en morceaux connus, et le préfixe `##` signale la suite d'un mot : un mot n'est donc presque jamais inconnu du modèle (le nombre de `[UNK]` est compté plus bas), alors que TF-IDF ignore tout mot absent de son vocabulaire. Le tokenizer ajoute `[CLS]` en tête, dont le vecteur final sert à la classification, et `[SEP]` en fin de phrase. Dans un lot, les phrases courtes sont complétées par `[PAD]` et le masque d'attention vaut 0 sur ces positions pour que l'attention les ignore.

# %%
if BERT_AVAILABLE:
    # ---- 6.1.a Découpage de quelques phrases ----
    phrases_demo = ["this film is n't bad at all .",
                    "an unflinchingly bleak and desperate vision .",
                    "the screenplay is self-congratulatory and overwrought .",
                    "embeddings"]
    display(pd.DataFrame([{"texte": s, "unités WordPiece": " ".join(tokenizer.tokenize(s)),
                           "mots": len(s.split()), "unités": len(tokenizer.tokenize(s))} for s in phrases_demo])
            .style.hide(axis="index"))

    # ---- 6.1.b Un lot de deux phrases : jetons spéciaux, rembourrage, masque d'attention ----
    lot = tokenizer(["a gorgeous film .", "the plot is thin , but the actors save it ."], padding=True, return_tensors="pt")
    toks = [tokenizer.convert_ids_to_tokens(ids) for ids in lot["input_ids"]]
    display(pd.DataFrame({"phrase 1": toks[0], "masque 1": lot["attention_mask"][0].tolist(),
                          "phrase 2": toks[1], "masque 2": lot["attention_mask"][1].tolist()}).T)

    # ---- 6.1.c Mesures sur la validation officielle ----
    mots_val = sorted({w for s in val_df.sentence for w in s.split()})
    n_unites = {w: len(tokenizer.tokenize(w)) for w in mots_val}
    hors_train = [w for w in mots_val if w.lower() not in train_vocab]
    ids_val = tokenizer(val_df.sentence.tolist(), add_special_tokens=False)["input_ids"]
    n_jetons = sum(len(seq) for seq in ids_val)
    TOKSTAT = {
        "mots_distincts_val": len(mots_val),
        "pct_mots_fragmentes": float(np.mean([n > 1 for n in n_unites.values()]) * 100),
        "mots_hors_train": len(hors_train),
        "pct_hors_train_fragmentes": float(np.mean([n_unites[w] > 1 for w in hors_train]) * 100) if hors_train else float("nan"),
        "jetons_inconnus_val": int(sum(i == tokenizer.unk_token_id for seq in ids_val for i in seq)),
        "unites_par_mot": n_jetons / sum(len(s.split()) for s in val_df.sentence),
        "p95_jetons": float(TOK["p95"]), "p99_jetons": float(TOK["p99"]), "max_jetons": int(TOK["max"]),
        "pct_tronques": float(TOK["trunc"]),
    }
    json.dump(TOKSTAT, open(OUT / "tokenisation_metrics.json", "w"), indent=2)
    exemples_rares = sorted(hors_train, key=lambda w: (-n_unites[w], w))[:8]
    display(pd.DataFrame({"mot absent du train": exemples_rares,
                          "unités WordPiece": [" ".join(tokenizer.tokenize(w)) for w in exemples_rares]}).style.hide(axis="index"))
    display(Markdown(
        f"Sur les {TOKSTAT['mots_distincts_val']} mots distincts de la validation, {fr(TOKSTAT['pct_mots_fragmentes'])} % sont "
        f"découpés en plusieurs unités ; c'est le cas de {fr(TOKSTAT['pct_hors_train_fragmentes'])} % des "
        f"{TOKSTAT['mots_hors_train']} mots absents du train. La validation compte {TOKSTAT['jetons_inconnus_val']} jeton(s) "
        f"`[UNK]` et {fr(TOKSTAT['unites_par_mot'], 2)} unités par mot en moyenne. Le texte le plus long mesuré en "
        f"section 6 fait {TOKSTAT['max_jetons']} jetons, donc `max_len = {MAX_LEN}` ne tronque "
        f"{'rien' if TOKSTAT['pct_tronques'] == 0 else fr(TOKSTAT['pct_tronques'], 2) + ' % des textes'} ; la valeur de 64 "
        f"proposée dans le cours {'suffirait aussi' if TOKSTAT['max_jetons'] <= 64 else 'tronquerait quelques textes'}."))
