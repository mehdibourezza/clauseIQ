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
3. Les fichiers de démo existent dans `samples/` :
   - `LohaCompanyltd_..._Supply_Agreement.txt` (sinon : `python -m scripts.export_cuad_samples --n 3`) ;
   - `Licence_CeCILL_V2.1-fr.txt`, un vrai contrat français public (sinon : le télécharger depuis
     https://cecill.info/licences/Licence_CeCILL_V2.1-fr.txt).
4. Le Wi-Fi sur place n'est pas garanti : tester une fois le dépôt en direct ; s'il échoue, rester sur les contrats pré-analysés.

Fermer tout le reste (notifications, onglets, `.env` jamais affiché à l'écran).

---

## Déroulé

| Temps | Écran | Ce que tu montres | Ce que tu dis |
|---|---|---|---|
| 0:00 | Liste des contrats | Les contrats déjà analysés, dont la licence CeCILL | « Un avocat dépose un contrat et voit, pour 41 types de clauses, la phrase exacte du contrat. » |
| 0:30 | Dépôt en direct (si le réseau répond) | `samples/Licence_CeCILL_V2.1-fr.txt`, bouton « Analyser le contrat » | « Un vrai contrat français, la licence CeCILL du CEA, du CNRS et de l'Inria. L'API répond tout de suite (202) ; l'analyse tourne en tâche de fond et la page redemande l'état toutes les 2 secondes. » Mesuré : 28 s et 0,12 $ pour ce contrat. Sans réseau : ouvrir la licence CeCILL déjà analysée. |
| 1:30 | Page du contrat français | Clic sur Governing Law (« Le Contrat est régi par la loi française. »), puis Cap On Liability | « Chaque passage surligné a été retrouvé mot pour mot dans le contrat. Le modèle peut se tromper, mais l'outil n'affiche jamais une citation qui n'existe pas. » |
| 2:30 | Même page | Clic sur Price Restrictions, puis le groupe « Non trouvées par le modèle » | « Ici, le passage existe, mais le classement est discutable : la vérification prouve que la phrase est dans le contrat, pas qu'elle est dans la bonne catégorie. Et "non trouvée" ne veut pas dire "absente". C'est l'avocat qui juge. » |
| 3:30 | Ligne d'informations sous le titre : « Citations rejetées 0 » | Puis, dans le terminal : `python -m pytest -k reworded` | « Sur 947 citations de l'évaluation, le modèle n'en a inventé aucune. Le vérificateur reste le filet de sécurité, et ce test le prouve : on lui donne une phrase reformulée, il la rejette et la compte. » |
| 4:30 | Terminal | `python -m pytest` | « 41 tests en une seconde. Ils remplacent Claude par un faux extracteur : aucun appel payant. Un test vérifie aussi que le cœur n'importe ni FastAPI, ni la base, ni Claude. » |
| 5:00 | Carte d'architecture | Schéma « Qui parle à qui » | Les trois décisions : modèle existant plutôt qu'entraînement, un appel pour 41 clauses, vérification mot pour mot. |
| 7:00 | README, section Evaluation | Tableau des résultats | « J'ai fixé la règle avant de mesurer et je l'ai testée avec un extracteur parfait, qui a révélé le défaut "Document Name". Résultat sur 20 contrats, deux passages : précision 0,63, rappel 0,78, identiques à 0,01 près. En analysant les erreurs, j'ai vu que les juristes annotent seulement la date, alors que le modèle cite la phrase qui la contient : ma règle était asymétrique. Je ne l'ai pas changée après coup ; l'analyse a posteriori, étiquetée comme telle, donne 0,72 et 0,90. Environ 25 s et 0,11 $ par contrat. » |
| 10:00 | Slide | Industrialisation | Ce qui manque et dans quel ordre : file d'attente, base gérée, authentification et droits par dossier, modèle hébergé dans l'UE, pipeline Azure DevOps exécuté, évaluation sur des contrats annotés par les avocats qui utiliseront l'outil (CUAD est en anglais, public depuis 2021, peut-être vu par le modèle ; un seul contrat français a été analysé). |

Mesuré le 12/10/2026 sur la licence CeCILL v2.1 (22 961 octets), par un dépôt dans l'interface : 28 s, 12 601 tokens en
entrée et 3 575 en sortie (0,12 $), 12 types de clauses, 19 citations, 19 retrouvées mot pour mot, 0 rejetée. Un seul
contrat : ce n'est pas une mesure de qualité en français.

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

Puis déposer `samples/Licence_CeCILL_V2.1-fr.txt` dans l'interface (environ 0,12 $ et 30 secondes).

Puis vérifier une fois, Wi-Fi coupé, que la liste et les pages de contrat s'affichent.
