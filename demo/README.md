# Démo en ligne (Hugging Face Spaces)

Une page publique où l'on tape une phrase et où BERT fine-tuné et la baseline TF-IDF donnent chacun leur
probabilité (calibrée par temperature scaling). Tout est gratuit : modèle sur le Hugging Face Hub, application
Gradio sur un Space « CPU basic ».

| Fichier | Rôle |
|---|---|
| `publier_modele_colab.py` | cellule à coller à la fin du notebook : publie le modèle, la baseline TF-IDF (en JSON) et les températures |
| `space/app.py` | l'application Gradio |
| `space/requirements.txt`, `space/README.md` | dépendances et configuration du Space |

## Étapes

1. **Compte et jeton.** Créer un compte sur huggingface.co (le nom d'utilisateur est supposé être `IlyasBBB` ;
   sinon, le changer dans `publier_modele_colab.py`, `space/app.py` et `space/README.md`). Dans Settings → Access
   Tokens, créer un jeton avec le droit **Write**.
2. **Réentraîner une fois dans Colab.** Ouvrir le notebook avec le badge Colab, choisir un GPU T4, mettre
   `SAVE_MODEL = True` dans la cellule de configuration (`RUN_EXTRA_SEEDS` peut rester à `False`), puis tout
   exécuter (environ 25 minutes).
3. **Publier le modèle.** Ajouter une cellule à la fin, y coller `publier_modele_colab.py`, l'exécuter et coller le
   jeton quand il est demandé. La cellule affiche l'adresse du modèle et l'accuracy mesurée pendant cette exécution
   (elle peut différer de quelques dixièmes des 92,66 % du rapport : l'entraînement sur GPU n'est pas parfaitement
   déterministe ; la fiche du modèle donne le chiffre réel).
4. **Créer le Space.** Sur huggingface.co : New → Space, nom `bert-sst2-demo`, SDK **Gradio**, matériel
   **CPU basic (gratuit)**, public. Puis Files → Add file → Upload files : déposer les trois fichiers de `space/`.
   Le Space se construit seul (quelques minutes) et s'ouvre sur `https://huggingface.co/spaces/IlyasBBB/bert-sst2-demo`.
5. **Relier.** Ajouter le lien de la démo au README du dépôt, au champ « Website » du dépôt GitHub, au projet
   LinkedIn et au CV Data et IA.

Un Space gratuit se met en veille après un moment sans visite ; la première ouverture suivante prend une minute.
