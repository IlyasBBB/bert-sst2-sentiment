# ---
# complement: J
# titre: Dix erreurs de BERT regroupées par cause (généré)
# ancre: a9728992
# ordre: 5
# actif: oui
# ---

# %% [markdown]
# ### 9.9 · Dix erreurs de BERT regroupées par cause
# Lecture faite par l'équipe sur les erreurs de l'exécution ci-dessus (`results/erreurs_bert_a_annoter.csv`). Les 64 erreurs de BERT ont toutes été lues et annotées. Répartition des causes : négation (9), ironie (11), avis mitigé (14), vocabulaire rare (6), étiquette discutable (18), autre (6). Le tableau en montre dix, au moins une par cause rencontrée ; les probabilités sont celles du fichier de résultats.
#
# | Cause | Phrase | Étiquette | P(positif) BERT | P(positif) TF-IDF | Lecture |
# |---|---|---|---|---|---|
# | négation | it 's everything you do n't go to the movies for . | négatif | 0,99 | 0,59 | Le « n't » retourne tout le sens ; BERT prédit pourtant positif. |
# | négation | ( d ) oes n't bother being as cloying or preachy as equivalent evangelical christian movies -- maybe the filmmakers know that the likely audience will already be among the faithful . | positif | 0,04 | 0,09 | La négation fait de « cloying » et « preachy » un compliment ; BERT prédit négatif. |
# | ironie | the lower your expectations , the more you 'll enjoy it . | négatif | 1,00 | 0,95 | Éloge empoisonné ; la prédiction suit le sens littéral de « enjoy it ». |
# | ironie | sit through this one , and you wo n't need a magic watch to stop time ; your dvd player will do it for you . | négatif | 1,00 | 0,98 | Le film serait si lent que le lecteur DVD s'arrête de lui-même. |
# | avis mitigé | it 's somewhat clumsy and too lethargically paced -- but its story about a mysterious creature with psychic abilities offers a solid build-up , a terrific climax , and some nice chills along the way . | négatif | 1,00 | 0,81 | Deux jugements autour de « but » : BERT prédit comme la seconde partie, l'étiquette suit la première. |
# | avis mitigé | the longer the movie goes , the worse it gets , but it 's actually pretty good in the first few minutes . | négatif | 0,99 | 0,83 | « the worse it gets » contre « pretty good » au début : BERT prédit comme la fin de la phrase. |
# | vocabulaire rare | holden caulfield did it better . | négatif | 0,96 | 0,46 | Tout repose sur Holden Caulfield, héros de « L'Attrape-cœurs » : sans la référence, la critique est invisible. |
# | étiquette discutable | moretti 's compelling anatomy of grief and the difficult process of adapting to loss . | négatif | 1,00 | 0,60 | « compelling » est positif et le deuil est le sujet ; l'étiquette négative se discute. |
# | étiquette discutable | while there 's something intrinsically funny about sir anthony hopkins saying ` get in the car , bitch , ' this jerry bruckheimer production has little else to offer | positif | 0,00 | 0,37 | La phrase finit sur « has little else to offer » ; l'étiquette positive se défend mal. |
# | autre | this one is definitely one to skip , even for horror movie fanatics . | négatif | 0,86 | 0,38 | Avis négatif sans détour, que BERT classe positif. |
