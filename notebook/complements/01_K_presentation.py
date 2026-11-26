# ---
# complement: K
# titre: Présentation des compléments
# ancre: 562fd8e0
# ordre: 1
# actif: oui
# ---

# %% [markdown]
# ### Compléments au travail initial
# Le travail initial (sections 1 à 10) est conservé tel quel. Les cellules ajoutées traitent les points du cours qui restaient ouverts. Chaque ajout suit le même protocole : sélection sur `dev`, validation officielle évaluée une seule fois par modèle, chiffres calculés à l'exécution.
#
# | Ajout | Raison | Section |
# |---|---|---|
# | Tokenisation WordPiece détaillée | étape 3 de la feuille de route du cours | 6.1 |
# | Fine-tuning répété sur trois graines | ne pas conclure sur une seule graine | 7 |
# | BERT gelé + régression logistique | bonus du cours : mesurer l'apport du fine-tuning complet | 7.1 |
# | DistilBERT (option, désactivée par défaut) | option CPU citée dans le cours | 7.2 |
# | Comparaison élargie et coût d'inférence | situer chaque modèle en accuracy et en coût | 8.1 |
# | Hypothèse H4 et écarts entre sous-groupes | hypothèse laissée ouverte en section 9 | 9.5 |
# | Phrases personnelles et paires minimales | étape 6 du cours, test contrôlé de H1 | 9.6 |
# | Calibration des probabilités | éclairer H3 et les erreurs très confiantes | 9.7 |
# | Dix erreurs regroupées par cause | livrable 4 du cours | 9.8 et 9.9 |
# | Synthèse reliée au MLM, au `[CLS]` et au fine-tuning | livrable 5 du cours | 10.3 et 10.4 |
#
# Avec ces ajouts, l'exécution complète dure environ 50 minutes sur un GPU T4 de Colab, dont environ 25 pour les deux graines supplémentaires.
