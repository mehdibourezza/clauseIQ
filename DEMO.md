# Démo ClauseIQ

Déroulé de la démonstration (environ 15 minutes), commandes de lancement et plan B.
Tout ce qui est montré ici a été exécuté au moins une fois ; les chiffres viennent de
`reports/` et de `PROJECT_REPORT.md`.

---

## Avant la démo (la veille, puis 30 minutes avant)

```powershell
# Terminal 1 : l'API (depuis C:\Users\boure\clauseiq)
conda activate clauseiq
uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 : l'interface
cd frontend
npx ng serve            # http://localhost:4200
```

Vérifications :

1. http://localhost:4200 affiche la liste des contrats pré-analysés (la démo marche sans réseau).
2. `python -m pytest` : tous les tests passent en environ une seconde.
3. Le fichier `samples/LohaCompanyltd_..._Supply_Agreement.txt` existe (sinon : `python -m scripts.export_cuad_samples --n 3`).
4. Le Wi-Fi sur place n'est pas garanti : tester une fois le dépôt en direct ; s'il échoue, rester sur les contrats pré-analysés.

Fermer tout le reste (notifications, onglets, `.env` jamais affiché à l'écran).

---

## Déroulé

| Temps | Écran | Ce que tu montres | Ce que tu dis |
|---|---|---|---|
| 0:00 | Liste des contrats | Les contrats déjà analysés | « Un avocat dépose un contrat et voit, pour 41 types de clauses, la phrase exacte du contrat. » |
| 0:30 | Dépôt en direct | `samples/Loha…Supply_Agreement.txt`, bouton Analyser | « L'API répond tout de suite (202) ; l'analyse tourne en tâche de fond et la page redemande l'état toutes les 2 secondes. » Mesuré : 16 s et environ 0,07 $ pour ce contrat. |
| 1:00 | Page du contrat | Clic sur Governing Law, puis Cap On Liability | « Chaque passage surligné a été retrouvé mot pour mot dans le contrat. Le modèle peut se tromper, mais l'outil n'affiche jamais une citation qui n'existe pas. » |
| 2:00 | Ligne d'informations sous le titre : « Citations rejetées 0 » | Puis, dans le terminal : `python -m pytest -k reworded` | « Sur 947 citations de l'évaluation, le modèle n'en a inventé aucune. Le vérificateur reste le filet de sécurité, et ce test le prouve : on lui donne une phrase reformulée, il la rejette et la compte. » |
| 3:00 | Terminal | `python -m pytest` | « 41 tests en une seconde. Ils remplacent Claude par un faux extracteur : aucun appel payant. Un test vérifie aussi que le cœur n'importe ni FastAPI, ni la base, ni Claude. » |
| 4:00 | Carte d'architecture | Schéma « Qui parle à qui » | Les trois décisions : modèle existant plutôt qu'entraînement, un appel pour 41 clauses, vérification mot pour mot. |
| 6:00 | README, section Evaluation | Tableau des résultats | « J'ai fixé la règle avant de mesurer et je l'ai testée avec un extracteur parfait, qui a révélé le défaut "Document Name". Résultat sur 20 contrats, deux passages : précision 0,63, rappel 0,78, identiques à 0,01 près. En analysant les erreurs, j'ai vu que les juristes annotent seulement la date, alors que le modèle cite la phrase qui la contient : ma règle était asymétrique. Je ne l'ai pas changée après coup ; l'analyse a posteriori, étiquetée comme telle, donne 0,72 et 0,90. Environ 25 s et 0,11 $ par contrat. » |
| 8:00 | Slide | Industrialisation | Ce qui manque et dans quel ordre : file d'attente, base gérée, authentification et droits par dossier, modèle hébergé dans l'UE, pipeline Azure DevOps exécuté, évaluation sur des contrats annotés par les avocats qui utiliseront l'outil (CUAD est en anglais, public depuis 2021, peut-être vu par le modèle). |

---

## Plan B

| Problème | Réaction |
|---|---|
| Pas de réseau | Le dépôt affiche « Could not reach the LLM API » dans la page du contrat : c'est la gestion d'erreur en direct. Continuer sur les contrats pré-analysés. |
| L'API ne démarre pas | Relancer `uvicorn` ; vérifier que l'environnement `clauseiq` est activé. |
| L'interface ne démarre pas | Port 4200 occupé : `npx ng serve --port 4300`. |
| Le portable lâche | Montrer la carte d'architecture (lien de l'artifact) et les captures d'écran des slides. |

---

## Option : montrer le découpage des contrats longs

Le découpage ne se déclenche qu'au-delà de 400 000 caractères, au-dessus de tout contrat de CUAD.
Pour le montrer, abaisser le seuil avant de lancer l'API :

```powershell
$env:LLM_MAX_CHARS_PER_CALL = "100000"
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Le plus long contrat de CUAD (338 211 caractères) part alors en 4 parties. Plus lent et plus cher qu'un
dépôt normal : à ne faire en direct que si on te le demande.

---

## Si la base de démo est vide

Les contrats pré-analysés vivent dans `clauseiq_review.db` (non versionné). Pour la recréer :

```powershell
python -m scripts.evaluate_llm --n 5 --save-to-db     # 5 contrats CUAD, environ 0,40 $ et 2 minutes (mesuré le 11/10/2026)
```

Puis vérifier une fois, Wi-Fi coupé, que la liste et les pages de contrat s'affichent.
