# ---
# complement: N
# titre: Chiffres des sections 3 et 4 enregistrés pour le rapport
# ancre: e20f4a8b
# ordre: 1
# actif: oui
# ---

# %%
# Chiffres déjà calculés en sections 3 et 4, enregistrés dans results/ pour que le rapport les cite
# sans recopie manuelle (aucun calcul nouveau, aucune décision prise ici). Convention : les clés « pct_ »
# sont en pourcentage (0 à 100) ; scripts/extraire_metriques.py les convertit en proportions « part_ ».
_etapes = list(clean_log.values())
DONNEES = {
    "train_officiel": len(train_full), "validation": len(val_df), "test_masque": int(N_TEST_HIDDEN),
    "doublons_exacts": int(dup), "phrases_contradictoires": int(n_conflict), "recouvrement_validation": int(overlap_val),
    "retires_contradictoires": _etapes[0] - _etapes[1], "retires_doublons": _etapes[1] - _etapes[2],
    "retires_recouvrement": _etapes[2] - _etapes[3], "train_nettoye": _etapes[3],
    "train_fit": len(train_fit), "dev": len(dev_df), "groupes_dev": len(dev_groups),
    "pct_positifs_train": float(train_full.label.mean() * 100), "pct_positifs_val": float(val_df.label.mean() * 100),
    "pct_positifs_train_fit": float(train_fit.label.mean() * 100), "pct_positifs_dev": float(dev_df.label.mean() * 100),
    "mots_moyens_train": float(words(train_full.sentence).mean()), "mots_moyens_val": float(words(val_df.sentence).mean()),
    "mots_medians_train": float(words(train_full.sentence).median()), "mots_medians_val": float(words(val_df.sentence).median()),
    "vocabulaire_train": len(train_vocab), "pct_types_val_hors_train": float(OOV_TYPES), "pct_occurrences_val_hors_train": float(OOV_TOK),
    "pct_contamination_groupes": float(LEAK["groupé"]), "pct_contamination_aleatoire": float(LEAK["aléatoire"]),
    "pct_negation_train": float(prev.iloc[0, 0]), "pct_negation_val": float(prev.iloc[1, 0]),
    "pct_opposition_train": float(prev.iloc[0, 1]), "pct_opposition_val": float(prev.iloc[1, 1]),
}
json.dump(DONNEES, open(OUT / "donnees_metrics.json", "w"), indent=2)

# Paramètres réellement utilisés par cette exécution (le bootstrap du rapport doit utiliser les mêmes)
PARAMS = {"seed": SEED, "n_boot": N_BOOT, "model_name": MODEL_NAME, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
          "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO, "max_len": MAX_LEN, "grad_clip": GRAD_CLIP,
          "dev_fraction": DEV_FRACTION, "max_train_samples": MAX_TRAIN_SAMPLES, "run_extra_seeds": RUN_EXTRA_SEEDS,
          "extra_seeds": EXTRA_SEEDS, "run_frozen": RUN_FROZEN, "run_distilbert": RUN_DISTILBERT,
          "frozen_max_train": FROZEN_MAX_TRAIN, "frozen_c_grid": FROZEN_C_GRID, "distil_model_name": DISTIL_MODEL_NAME,
          "confident": CONFIDENT, "n_calib_bins": N_CALIB_BINS,
          "target_val_acc": TARGET_VAL_ACC, "device": device, "amp": str(amp_dtype).split(".")[-1] if use_amp else "aucune",
          "execution_complete": MODEL_NAME == "bert-base-uncased" and MAX_TRAIN_SAMPLES is None,
          "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "aucun",
          "versions": {"python": platform.python_version(), **{m: ver(m) for m in ["torch", "transformers", "datasets", "sklearn", "scipy", "pandas", "numpy"]}}}
json.dump(PARAMS, open(OUT / "parametres_metrics.json", "w"), indent=2)
print(f"{len(DONNEES)} chiffres des sections 3-4 et {len(PARAMS)} paramètres enregistrés dans results/")
