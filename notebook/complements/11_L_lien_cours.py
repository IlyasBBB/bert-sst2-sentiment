# ---
# complement: L
# titre: Lien avec le cours (MLM, [CLS], fine-tuning)
# ancre: 6f9106cb
# ordre: 2
# actif: oui
# ---

# %% [markdown]
# ### 10.4 · Lien avec le cours : pré-entraînement, jeton `[CLS]` et fine-tuning
# BERT arrive sur SST-2 avec ce qu'il a appris en devinant des mots masqués dans 3,3 milliards de mots de texte (MLM). Un mot absent du train n'est donc pas inconnu pour lui : WordPiece le découpe en unités vues pendant le pré-entraînement (section 6.1). Cela ne se traduit pas par un avantage mesurable : en section 9.5, l'écart entre BERT et la baseline n'est pas plus grand sur les phrases qui contiennent un mot hors vocabulaire, et H4 n'est pas confirmée. Le vecteur `[CLS]` du modèle pré-entraîné n'a été entraîné sur aucun objectif de sentiment : pendant le pré-entraînement, il sert seulement à prédire si deux phrases se suivent. Les vecteurs du BERT gelé, utilisés tels quels avec une régression logistique, font déjà mieux que TF-IDF, mais restent nettement sous la version fine-tunée. Pour la variante retenue sur `dev` (moyenne des jetons), les intervalles de confiance des deux écarts excluent 0 (sections 7.1 et 8.1). Le fine-tuning n'ajoute que 1 538 paramètres, une couche linéaire de 768 × 2 poids et 2 biais, mais il ajuste aussi les 110 M de poids du réseau avec un petit taux d'apprentissage (2 × 10⁻⁵). L'écart entre version gelée et version fine-tunée estime ce que rapporte cette adaptation de tout le réseau, avec une réserve : le classifieur posé sur les vecteurs n'est pas tout à fait le même dans les deux cas. Parmi les erreurs qui restent (section 9.9), les plus nombreuses viennent d'une étiquette discutable, d'un avis mitigé ou d'une ironie, des cas où la phrase seule ne suffit pas toujours à trancher.
