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
| 0:30 | Dépôt en direct | `samples/Loha…Supply_Agreement.txt`, bouton Analyser | « L'API répond tout de suite (202) ; l'analyse tourne en tâche de fond et la page redemande l'état toutes les 2 secondes. » Mesuré : environ 16 s pour ce contrat. |
| 1:00 | Page du contrat | Clic sur Governing Law, puis Cap On Liability | « Chaque passage surligné a été retrouvé mot pour mot dans le contrat. Le modèle peut se tromper, mais l'outil n'affiche jamais une citation qui n'existe pas. » |
| 2:00 | Contrat avec une citation rejetée | Clic sur la clause concernée : encadré orange | « Ici le modèle a cité un texte introuvable : il n'est pas surligné, mais il est compté. » (contrat à choisir après l'évaluation) |
| 3:00 | Terminal | `python -m pytest` | « Les tests remplacent Claude par un faux extracteur : aucun appel payant. Un test vérifie aussi que le cœur n'importe ni FastAPI, ni la base, ni Claude. » |
| 4:00 | Carte d'architecture | Schéma « Qui parle à qui » | Les trois décisions : modèle existant plutôt qu'entraînement, un appel pour 41 clauses, vérification mot pour mot. |
| 6:00 | `reports/` ou README | Résultats de l'évaluation | « J'ai vérifié la mesure avant de mesurer le modèle » (le défaut « Document Name » trouvé et corrigé), puis les chiffres. |
| 8:00 | Slide | Industrialisation | Ce qui manque et dans quel ordre : file d'attente, base gérée, authentification et droits par dossier, modèle hébergé dans l'UE, pipeline Azure DevOps exécuté, évaluation sur des contrats annotés par les avocats qui utiliseront l'outil. |

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
