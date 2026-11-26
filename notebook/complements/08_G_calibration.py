# ---
# complement: G
# titre: Calibration des probabilités (H3)
# ancre: a9728992
# ordre: 3
# actif: oui
# ---

# %% [markdown]
# ### 9.7 · Calibration des probabilités
# La section 9.4 a compté les erreurs de BERT commises avec une probabilité supérieure à 0,9 ou inférieure à 0,1. Reste à savoir si ces probabilités veulent dire quelque chose. Un modèle bien calibré qui annonce 95 % de confiance a raison 95 fois sur 100. Le diagramme de fiabilité compare, tranche par tranche de confiance, ce que le modèle annonce et ce qu'il obtient. L'erreur de calibration attendue (ECE) résume cet écart et le score de Brier mesure la qualité globale des probabilités. On essaie aussi le *temperature scaling* : un seul paramètre, ajusté sur `dev`, qui adoucit ou durcit les probabilités sans changer aucune prédiction. `dev` contient des syntagmes et la validation des phrases complètes, ce qui peut limiter le transfert de ce réglage.

# %%
from scipy.optimize import minimize_scalar

def calibration(y, p, n_bins=N_CALIB_BINS):
    """ECE (confiance de la classe prédite, tranches de largeur égale sur [0,5 ; 1]), Brier, table de fiabilité."""
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    conf = np.maximum(p, 1 - p); ok = ((p >= .5).astype(int) == y)
    bornes = np.linspace(.5, 1, n_bins + 1); t = np.clip(np.digitize(conf, bornes[1:-1]), 0, n_bins - 1)
    ece, rows_ = 0.0, []
    for b in range(n_bins):
        sel = t == b
        if sel.any():
            ece += sel.mean() * abs(ok[sel].mean() - conf[sel].mean())
            rows_.append({"tranche": f"[{bornes[b]:.2f} ; {bornes[b + 1]:.2f}]", "n": int(sel.sum()),
                          "confiance moyenne": conf[sel].mean(), "accuracy": ok[sel].mean()})
    return float(ece), float(np.mean((p - y) ** 2)), pd.DataFrame(rows_)

def fit_temperature(y, p):
    """Température T qui minimise la log-vraisemblance négative de sigmoïde(logit / T) ; T > 1 adoucit."""
    y = np.asarray(y); z = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    nll = lambda logt: -np.mean(y * -np.logaddexp(0, -z / np.exp(logt)) + (1 - y) * -np.logaddexp(0, z / np.exp(logt)))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))

def apply_temperature(p, temp):
    z = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    return 1 / (1 + np.exp(-z / temp))

CALIB, courbes = {}, {}
for m in MODELS:
    dev_p = pd.read_csv(OUT / f"{m}_dev_preds.csv"); val_p = pd.read_csv(OUT / f"{m}_val_preds.csv")
    ece_, brier_, table_ = calibration(val_p.label, val_p.proba)
    temp_ = fit_temperature(dev_p.label.values, dev_p.proba.values)
    p_t = apply_temperature(val_p.proba.values, temp_)
    ece_t, brier_t, table_t = calibration(val_p.label, p_t)
    err_ = (val_p.pred != val_p.label).values
    conf_err = np.maximum(val_p.proba.values, 1 - val_p.proba.values)[err_]
    CALIB[m] = {"ece": ece_, "brier": brier_, "temperature": temp_, "ece_apres_T": ece_t, "brier_apres_T": brier_t,
                "pct_erreurs_confiantes": float((conf_err >= CONFIDENT).mean() * 100) if err_.any() else float("nan"),
                "confiance_moyenne_erreurs": float(conf_err.mean()) if err_.any() else float("nan")}
    courbes[m] = (table_, table_t)
json.dump(CALIB, open(OUT / "calibration_metrics.json", "w"), indent=2)
display(pd.DataFrame(CALIB).T.rename(index=LAB).rename(columns={
    "ece": "ECE", "brier": "Brier", "temperature": "température T", "ece_apres_T": "ECE après T", "brier_apres_T": "Brier après T",
    "pct_erreurs_confiantes": "% d'erreurs confiantes", "confiance_moyenne_erreurs": "confiance moyenne des erreurs"})
    .style.format({"ECE": "{:.3f}", "Brier": "{:.3f}", "température T": "{:.2f}", "ECE après T": "{:.3f}", "Brier après T": "{:.3f}",
                   "% d'erreurs confiantes": "{:.1f}", "confiance moyenne des erreurs": "{:.3f}"}))

fig, axes = plt.subplots(1, len(MODELS), figsize=(5.2 * len(MODELS), 4.2), squeeze=False)
for ax, m, col in zip(axes[0], MODELS, [BLUE, ORANGE]):
    tb, tbt = courbes[m]
    ax.plot([.5, 1], [.5, 1], "k:", lw=1, label="calibration parfaite")
    ax.plot(tb["confiance moyenne"], tb["accuracy"], "o-", color=col, label=f"probabilités brutes (ECE {CALIB[m]['ece']:.3f})")
    ax.plot(tbt["confiance moyenne"], tbt["accuracy"], "s--", color="grey", alpha=.8,
            label=f"après temperature scaling, T = {CALIB[m]['temperature']:.2f} (ECE {CALIB[m]['ece_apres_T']:.3f})")
    ax.set(title=f"{LAB[m]} : diagramme de fiabilité (validation)", xlabel="confiance annoncée", ylabel="accuracy observée",
           xlim=(.5, 1.01), ylim=(.3, 1.01)); ax.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "calibration.png", dpi=130); plt.show()
