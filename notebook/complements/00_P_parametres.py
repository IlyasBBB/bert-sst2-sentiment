# ---
# complement: P
# titre: Paramètres des compléments
# ancre: 024996e2
# ordre: 1
# actif: oui
# ---

# %% [markdown]
# Réglages propres aux compléments, regroupés juste après la configuration d'origine ; les cellules du travail initial ne changent pas. `RUN_EXTRA_SEEDS` passe à `True` pour répéter le fine-tuning avec deux autres graines (environ 12 minutes de plus par graine sur T4) et mesurer la part du hasard de l'entraînement dans les résultats.

# %%
# >>> PARAMÈTRES DES COMPLÉMENTS
# ecrase-volontairement: RUN_EXTRA_SEEDS
RUN_EXTRA_SEEDS   = True                        # section 7 : variabilité inter-graines (False dans la version initiale)
RUN_FROZEN        = True                        # section 7.1 : BERT gelé + régression logistique (bonus du cours)
FROZEN_MAX_TRAIN  = None                        # None = tout train_fit ; un entier = sous-échantillon
FROZEN_C_GRID     = [0.01, 0.1, 1, 10]          # C de la régression logistique, choisi sur dev
RUN_DISTILBERT    = False                       # section 7.2 : option CPU du cours (~7 min de plus sur T4)
DISTIL_MODEL_NAME = "distilbert-base-uncased"
N_CALIB_BINS      = 10                          # section 9.7 : tranches du diagramme de fiabilité
CONFIDENT         = 0.9                         # seuil des erreurs « confiantes », comme en section 9.4

fr = lambda x, d=1: f"{x:.{d}f}".replace(".", ",")                     # 0.5 -> "0,5" dans les textes générés
fr_sg = lambda x, d=1: ("+" if x >= 0 else "−") + fr(abs(x), d)            # nombre signé avec un vrai signe moins
fr_pts = lambda x, d=1: f"{fr_sg(x, d)} point{'s' if abs(x) >= 2 else ''}"   # écart signé en points de pourcentage
fr_pt = lambda x, d=1: f"{fr(x, d)} point{'s' if abs(x) >= 2 else ''}"    # même chose, sans signe

from contextlib import contextmanager
@contextmanager
def silence_hf():
    """Masque le rapport de chargement de Hugging Face (expliqué en section 6) lors des rechargements de modèles."""
    from transformers.utils import logging as hf_logging
    niveau = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
    try:
        yield
    finally:
        hf_logging.set_verbosity(niveau); hf_logging.enable_progress_bar()
