# ---
# complement: I
# titre: Synthèse calculée des compléments
# ancre: 6f9106cb
# ordre: 1
# actif: oui
# ---

# %% [markdown]
# ### 10.3 · Ce que les compléments ajoutent
# Comme la conclusion 10.2, ce texte est produit à partir des fichiers de cette exécution : aucun chiffre n'y est écrit à la main.

# %%
def _lire(nom):
    f = OUT / nom
    return json.load(open(f)) if f.exists() else None

pc = lambda x, d=1: fr(100 * x, d) + " %"
L2 = []
if (OUT / "bert_seeds.csv").exists():
    sd_ = pd.read_csv(OUT / "bert_seeds.csv")
    tous = sd_.val_acc.min() >= TARGET_VAL_ACC - .005
    L2.append(f"**Graines.** Sur {len(sd_)} graines, BERT obtient en moyenne {pc(sd_.val_acc.mean())} d'accuracy de validation "
              f"(écart-type {fr_pt(100 * sd_.val_acc.std(ddof=1))}, de {pc(sd_.val_acc.min())} à {pc(sd_.val_acc.max())}). "
              f"L'objectif d'environ {pc(TARGET_VAL_ACC, 0)} est {'atteint avec chaque graine' if tous else 'manqué avec au moins une graine'}.")
fc, fmn, fb, ft = _lire("frozen_cls_metrics.json"), _lire("frozen_mean_metrics.json"), _lire("bert_metrics.json"), _lire("tfidf_metrics.json")
if fc and fmn and fb and ft:
    gele = max([fc, fmn], key=lambda r: r["dev"]["accuracy"])          # variante gelée choisie sur dev
    gain_ft = fb["val"]["accuracy"] - gele["val"]["accuracy"]
    paires_ = {(p_["a"], p_["b"]): p_ for p_ in (_lire("comparaisons_appariees.json") or [])}
    ic_txt = lambda p_: (f" (IC à 95 % de l'écart [{fr_sg(100 * p_['ic_bas'])} ; {fr_sg(100 * p_['ic_haut'])}] points, "
                         f"{'qui exclut 0' if p_['ic_bas'] > 0 or p_['ic_haut'] < 0 else 'qui contient 0'})") if p_ else ""
    pft = next((v for (a_, b_), v in paires_.items() if b_ == "bert" and a_.startswith("frozen")), None)
    pbg = next((v for (a_, b_), v in paires_.items() if a_ == "tfidf" and b_.startswith("frozen")), None)
    ic_ft, ic_bg = ic_txt(pft), ic_txt(pbg)
    L2.append(f"**BERT gelé.** Sans fine-tuning, une régression logistique sur les vecteurs de BERT atteint {pc(fc['val']['accuracy'])} "
              f"avec le jeton `[CLS]` et {pc(fmn['val']['accuracy'])} avec la moyenne des jetons, contre {pc(fb['val']['accuracy'])} "
              f"après fine-tuning complet et {pc(ft['val']['accuracy'])} pour TF-IDF. La variante gelée retenue sur `dev` "
              f"{'dépasse' if gele['val']['accuracy'] > ft['val']['accuracy'] else 'reste sous'} la baseline{ic_bg}, et le fine-tuning complet "
              + (f"lui ajoute {fr_pts(100 * gain_ft)}{ic_ft}." if gain_ft > 0
                 else "ne fait pas mieux qu'elle, ce qui signale un problème d'entraînement à examiner."))
dd = OUT / "ecart_des_ecarts.csv"
if dd.exists():
    DD_ = pd.read_csv(dd)
    morceaux = [f"{r['marqueur']} {fr_pts(100 * r['différence'])} [{fr_sg(100 * r['IC95 bas'])} ; {fr_sg(100 * r['IC95 haut'])}]"
                for _, r in DD_.iterrows()]
    etablis = DD_[DD_["IC95 bas"] > 0].marqueur.tolist()
    L2.append("**Hypothèses H1, H2 et H4.** Avantage de BERT dans le groupe moins avantage hors du groupe : "
              + " ; ".join(morceaux) + ". "
              + (f"La différence est établie au seuil de 95 % pour : {', '.join(etablis)}." if etablis
                 else "Aucune de ces différences n'est établie au seuil de 95 %."))
mp = _lire("paires_minimales_metrics.json")
if mp and "bert|inversion négation « not »" in mp:
    opp = lambda m: (mp[f"{m}|opposition « + but − »"] + mp[f"{m}|opposition « − but + »"]) / 2
    L2.append(f"**Paires minimales.** Ajouter « not » inverse la prédiction de BERT dans {pc(mp['bert|inversion négation « not »'], 0)} "
              f"des paires, contre {pc(mp['tfidf|inversion négation « not »'], 0)} pour TF-IDF. Dans les phrases « A, but B », "
              f"BERT suit l'avis de la seconde proposition dans {pc(opp('bert'), 0)} des cas, TF-IDF dans {pc(opp('tfidf'), 0)}.")
cal = _lire("calibration_metrics.json")
if cal and "bert" in cal:
    cb = cal["bert"]
    L2.append(f"**Calibration.** L'ECE de BERT vaut {fr(cb['ece'], 3)}, contre {fr(cal['tfidf']['ece'], 3)} pour TF-IDF. "
              f"Ses erreurs sont commises avec une confiance moyenne de {pc(cb['confiance_moyenne_erreurs'])}. "
              f"Le temperature scaling ajusté sur `dev` (T = {fr(cb['temperature'], 2)}) "
              f"{'ramène' if cb['ece_apres_T'] < cb['ece'] else 'porte'} l'ECE de validation à {fr(cb['ece_apres_T'], 3)}.")
co = _lire("cout_metrics.json")
if co and "bert_s" in co:
    L2.append(f"**Coût.** Prédire les 872 phrases prend {fr(co['bert_s'], 2)} s à BERT sur {co['device'].upper()} et "
              f"{fr(co['tfidf_s'], 3)} s à TF-IDF sur CPU ; BERT compte {fr(co['bert_params'] / 1e6, 0)} millions de paramètres, "
              f"la baseline {fr(co['tfidf_params'] / 1e3, 0)} milliers.")
display(Markdown("\n\n".join(L2) if L2 else "Aucun complément n'a produit de résultat dans cette exécution."))
