# 5MLDE-project

Mini projet du module 5MLDE — énoncé : [subject.md](subject.md).

## Prérequis

- Docker (Compose v2)
- [uv](https://docs.astral.sh/uv/) — Python 3.12 est installé automatiquement
- un clone git du dépôt : la version du paquet est calculée depuis les tags git

## Démarrage rapide

Tout démarrer, d'une machine vide à un système qui prédit seul :

```powershell
copy .env.example .env
docker compose up -d --build --wait                                       # stack + migrations du schéma
uv sync
uv run python -m connectors.health                                        # toutes les briques répondent
docker compose exec worker prefect deployment run ingest-candles/horaire  # historique Binance (archives)
docker compose exec worker prefect deployment run train-volatility/hebdomadaire
docker compose exec worker prefect deployment run train-returns/hebdomadaire-rendement
uv run python -m registry statut volatilite                               # un champion existe ?
uv run python -m registry statut rendement
```

Si l'entraînement ne passe pas la règle de promotion, il n'y a pas de champion : promouvoir la version
à la main (`uv run python -m registry promouvoir <modele> --version N --raison "..."`, voir
[Service en production](#service-en-production)). Seule une image construite depuis un tag propre publie
au registre.

Ce que le système fait ensuite tout seul (worker Prefect, plannings en UTC) :

- chaque heure : ingestion des bougies (`:02`), prévisions `volatilite` (`:07`) et `rendement` (`:08`),
  retours sur les horizons clôturés, santé quotidienne et retour arrière automatique (`:20`, `:21`) ;
- chaque lundi : réentraînement des deux modèles et règle de promotion (03:00, 03:30) ;
- sans champion, le flow de prévision se termine en `cancelled` avec la raison tracée, sans planter ;
- l'interrupteur (page Service de l'application) arrête tous les flows planifiés jusqu'à la remise en
  marche ; les bascules sont tracées et annotées dans Grafana ;
- un lot de bougies invalide est rejeté (run en échec, alerte Grafana) ; une brique indisponible rend
  l'API `/health` dégradée (503) et les flows reprennent au run suivant.

Où regarder : application http://localhost:8501 (pilotage), Grafana http://localhost:3000 (supervision),
Prefect http://localhost:4200 (runs), MLflow http://localhost:5000 (expériences et registre).
Arrêt : `docker compose down` (les données restent dans les volumes).

## Stack locale (Docker Compose)

Copie le fichier d'environnement et adapte-le si nécessaire (ports, mots de passe, adresses), puis démarre la stack :

```powershell
copy .env.example .env
docker compose up -d --build --wait
```

Pour arrêter la stack, lance `docker compose down`. L'option `-v` supprime aussi les volumes et leurs données.

| Service | URL (hôte) | Rôle |
|---|---|---|
| MLflow 3.16.1 | http://localhost:5000 | Expériences, registre de modèles, artefacts (backend PostgreSQL, base `mlflow`) |
| Prefect 3.8.7 | http://localhost:4200 | Orchestration des flows (backend PostgreSQL, base `prefect`) |
| PostgreSQL 18.6 | `localhost:5432` | Bases `mlops` (schéma `tracing`), `prefect`, `mlflow` |
| Grafana 13.2.3 | http://localhost:3000 | Supervision (admin / `5mlde-project-admin` par défaut) |
| API du projet | http://localhost:8000/docs | FastAPI : santé, prévisions, registre, marche/arrêt (image du projet) |
| Application | http://localhost:8501 | Streamlit de pilotage, cliente de l'API uniquement (image du projet) |
| Worker | — | Sert les déploiements Prefect planifiés (image du projet) |

Tous les ports sont liés à `127.0.0.1` uniquement et se changent dans `.env`.
Les mots de passe PostgreSQL ne sont appliqués qu'à la création du volume `postgres_data`.

### Rôles PostgreSQL

Créés par `infra/postgres/init/001-roles.sql` au premier démarrage du volume ; les tables de
traçabilité sont créées par `infra/postgres/init/002-tracing.sql`.

Sur un volume existant, le service ponctuel `postgres-migrate` applique à chaque `docker compose up`
les scripts `init/*.sql` pas encore passés (suivi dans `public.schema_migrations`, voir
`infra/postgres/migrate.sh`). Pour faire évoluer le schéma, ajouter un nouveau script numéroté
(`007-...sql`) sans modifier ceux déjà publiés : pas besoin de `docker compose down -v`.

| Rôle | Accès |
|---|---|
| `api_writer` | `mlops.tracing` : SELECT/INSERT (traces en ajout seul) |
| `grafana_reader` | lecture seule (`default_transaction_read_only`, timeout 10 s) sur `mlops.tracing`, `prefect` et `mlflow` |
| `prefect` | propriétaire de la base `prefect` (migrations gérées par le serveur Prefect) |
| `mlflow` | propriétaire de la base `mlflow` (migrations gérées par le serveur MLflow) |

Grafana est provisionné automatiquement : trois datasources (`projet-traces`, `projet-prefect`,
`projet-mlflow`) et les dashboards JSON placés dans `infra/grafana/dashboards/`.

## Paquet `connectors`

Un connecteur par brique de la stack, configuré uniquement par variables d'environnement
(`.env` à la racine, voir `.env.example`). Une configuration invalide lève `ConfigError` ; un
service injoignable ou qui refuse la connexion lève `ConnectorError` avec un conseil de correction.

| Module | Rôle |
|---|---|
| `connectors.config` | Lecture et validation de la configuration de chaque service |
| `connectors.mlflow` | Runs, log et enregistrement des modèles, alias, chargement |
| `connectors.prefect` | Santé, déclenchement de déploiements, contexte de run, artifacts |
| `connectors.postgres` | Écriture des traces dans `mlops.tracing` (pool `api_writer`) |
| `connectors.grafana` | Santé, datasources, dashboards, annotations |
| `lineage.provenance` | Version et empreintes du code et des dépendances |
| `connectors.health` | Vérification de toutes les briques |

```powershell
uv sync
uv run python -m connectors.health
uv run python -m connectors.health mlflow postgres
```

La première commande vérifie toutes les briques ; la seconde limite le contrôle à MLflow et PostgreSQL.

### Traçabilité des modèles

Chaque run et chaque modèle MLflow porte la provenance du code qui l'a produit (tags `code.*`) :
version calculée depuis git (`0.2.0` = tag `v0.2.0`, `0.2.1.devN+g<sha>` = commit, suffixe
`.d<date>` = arbre modifié), empreinte SHA-256 des paquets du projet (`api`, `app`, `connectors`,
`features`, `flows`, `ingestion`, `lineage`, `marketdata`, `registry`, `serving`, `strategies` et
`training`) et des dépendances installées. Le code des paquets et la liste exacte des dépendances sont
embarqués dans le modèle. Le run reçoit aussi `provenance/code.zip` (archive du code exécuté, mêmes
fichiers que l'empreinte `code.sha256`) et `provenance/requirements.txt`, ainsi que les métriques
système (`system/*` : CPU, mémoire, disque, réseau, échantillonnées toutes les 2 s).

Le run d'entraînement `volatilite` joint aussi des graphiques dans `graphiques/` : courbe
d'apprentissage de `hgb` (perte entraînement/validation et QLIKE par itération, dernier fold),
QLIKE par fold walk-forward, scores et QLIKE hebdomadaire sur le test final, volatilité prévue contre
réalisée par symbole, nuage prévu/réalisé, valeur du portefeuille de chaque stratégie et importance des
features par permutation.

`register_model` refuse un modèle dont le code n'est pas identifiable de façon fiable : arbre Git
modifié ou installation éditable. Pour produire un modèle publiable, crée un tag propre puis lance
l'entraînement depuis une installation non éditable :

```powershell
git tag v0.2.0
uv run --no-editable python -m <votre_module>
```

## Paquet `marketdata`

Définition des données : bougies horaires Binance spot pour `BTCUSDT`, `ETHUSDT`, `SOLUSDT`,
`BNBUSDT` et `XRPUSDT`, depuis 2022-01. Aucun accès réseau dans ce paquet.

| Module | Rôle |
|---|---|
| `marketdata.universe` | Symboles, intervalle (`1h`), début de l'historique |
| `marketdata.sources` | URLs des archives mensuelles et de l'API REST, lecture du `.CHECKSUM`, normalisation (horodatages ms ou µs → UTC à la µs) |
| `marketdata.schema` | Schéma pandera d'une bougie normalisée (types, unités, bornes) |
| `marketdata.quality` | Règles qualité : bloquantes (schéma, doublons, alignement, OHLC, volumes, variation > 50 %) ou signalées (bougies manquantes jusqu'à 1 %, variation > 20 %) |
| `marketdata.identity` | Identité d'un jeu de données : symboles, période, lignes, SHA-256 du contenu (tags `data.*`) |

Les archives historiques sont téléchargées depuis [`data.binance.vision`](https://data.binance.vision/) et
les bougies récentes depuis l'API REST Binance Spot (`/api/v3/klines`). Consulter les [conditions
d'utilisation de Binance](https://www.binance.com/en/terms) (date d'accès à la référence : 2026-11-18) et
vérifier leur version applicable avant toute redistribution. Ce projet ne redistribue pas les données.
Les fichiers téléchargés sont conservés localement dans `data/`, ignoré par Git : aucune archive ni
bougie brute n'est versionnée dans le dépôt.

## Paquet `features`

Le calcul des variables et des cibles est séparé dans `features.build` et `features.targets`.
L'API publique reste disponible depuis le paquet :
`from features import build_features, target_return, target_volatility`.

## Paquet `training`

`training.dataset.prepare_return_inputs` contrôle et identifie le jeu de bougies, puis calcule séparément
les features et la cible 4 h au même instant de référence. `training.dataset.prepare_return_dataset`
les joint sur `(symbol, open_time)` et retire les cibles ou features incomplètes ; le résultat conserve
l'identité des données et le rapport qualité.
`training.splits.walk_forward_splits` produit des indices positionnels pour des folds expansifs :
12 mois initiaux, validations mensuelles et trois derniers mois réservés au test final.
`training.splits.holdout_split` renvoie les indices positionnels d'entraînement et de ce test final.
Les deux fonctions appliquent un embargo au moins égal à l'horizon et excluent les lignes sans cible
si `target_column` est fourni. `training.metrics.information_coefficient` calcule l'IC de Spearman
transversal par horodatage ; `training.metrics.icir` calcule l'ICIR d'une série d'IC.

### Modèle `volatilite`

`training.volatility.fit_volatility_model` compare Ridge et HistGradientBoosting à trois références :
la persistance (volatilité des dernières 24 h), la volatilité de la semaine précédente ramenée à 24 h,
et la moyenne géométrique de la volatilité cible par symbole, calculée sur l'entraînement. Le candidat
retenu est celui dont le QLIKE moyen est le plus faible sur les folds walk-forward. Il est réentraîné
hors test final, puis évalué une seule fois sur les trois derniers mois réservés.

La promotion exige deux niveaux : (1) un QLIKE meilleur que celui de la meilleure référence sur le test
final ; (2) un Sharpe au moins égal à celui de la stratégie de ciblage de volatilité pilotée par la
persistance, frais inclus.

### Modèle `rendement`

`training.returns.fit_return_model` prédit le rendement logarithmique des quatre prochaines heures.
Ridge et HistGradientBoosting sont comparés en walk-forward ; le candidat ayant le meilleur IC moyen
sur les folds est réentraîné hors test final, puis évalué une seule fois sur les trois derniers mois.
Les références sont le rendement nul et les momentum à 1 h et 4 h.

L'IC est la corrélation de Spearman transversale entre prédictions et rendements réalisés, calculée par
heure ; l'ICIR rapporte la moyenne des IC à leur écart-type d'échantillon (sans annualisation). La
promotion vérifie deux niveaux : (1) l'IC du candidat sur le test final doit battre la meilleure
référence dont l'IC est défini ; (2) son Sharpe long/short, net de frais, doit battre l'équipondéré brut
et les références momentum long/short, et ne pas être inférieur à celui du champion existant.
La stratégie long/cash est aussi évaluée et journalisée, mais ne constitue pas le critère de promotion.

Pour entraîner et journaliser le modèle, puis éventuellement appliquer la règle de promotion :

```powershell
uv run python -m training.returns
```

Pour tenter la promotion au registre, lance l'entraînement depuis un tag propre et une installation
non éditable :

```powershell
uv run --no-editable python -m training.returns --registre
```

La santé en production est mesurée sur une fenêtre glissante de sept jours : une version rendement est
dégradée si son IC de Spearman est négatif ou si son PnL paper est négatif. Après trois jours consécutifs
de santé dégradée, le flow tente de rétablir le champion précédent.

## Service en production

`worker` et `api` tournent sur l'image du projet (`Dockerfile`) : le paquet y est installé en copie
non éditable avec les dépendances de `uv.lock`, et sa version est calculée depuis Git au build.
Seule une image construite depuis un tag propre (par exemple `git tag v0.2.0`, puis
`docker compose build worker api`) peut enregistrer des modèles au registre ; sinon l'entraînement est
loggé dans MLflow mais non publié.

| Déploiement Prefect | Planning (UTC) | Rôle |
|---|---|---|
| `ingest-candles/horaire` | `2 * * * *` | Bougies de l'heure écoulée (API REST), archives mensuelles manquantes |
| `predict-volatility/horaire-prevision` | `7 * * * *` | Prévision 24 h du `champion` (et du `challenger`), tracée dans `tracing.predictions` |
| `predict-returns/horaire-rendement` | `8 * * * *` | Prévision rendement 4 h du `champion` (et du `challenger`) |
| `feedback-volatility/horaire-retours` | `20 * * * *` | Retours à horizon clôturé, santé quotidienne, retour arrière automatique |
| `feedback-returns/horaire-retours-rendement` | `21 * * * *` | Retours rendement, IC et PnL paper, retour arrière automatique |
| `train-returns/hebdomadaire-rendement` | `0 3 * * 1` | Réentraînement rendement et règle de promotion |
| `train-volatility/hebdomadaire` | `30 3 * * 1` | Réentraînement et règle de promotion (niveaux 1 et 2) |

Un déploiement se lance aussi à la main (« Quick run » dans l'UI Prefect, ou
`docker compose exec worker prefect deployment run predict-volatility/horaire-prevision`).
Hors conteneur, les commandes par défaut ciblent `volatilite`. Passe `--model rendement` pour choisir le
modèle de rendement :

```powershell
uv run python -m flows.predict --model rendement
uv run python -m flows.feedback --model rendement
uv run python -m flows.train --model rendement
```

Registre et service en ligne de commande :

```powershell
uv run python -m registry statut volatilite
uv run python -m registry promouvoir volatilite --version 2 --raison "..."
uv run python -m registry retour-arriere volatilite
uv run python -m registry statut rendement
uv run python -m registry promouvoir rendement --version 2 --raison "..."
uv run python -m registry retour-arriere rendement
```

La promotion manuelle exige une raison et est tracée dans le registre.

`POST /system/state` (ou l'interrupteur de l'application) arrête les flows planifiés : ils se
terminent en `cancelled` dans `tracing.pipeline_runs`, sans rien écrire, jusqu'à la remise en marche.

## Supervision (Grafana)

Quatre dashboards, dossier « Projet 5MLDE » (http://localhost:3000), versionnés dans
`infra/grafana/dashboards/` et relus toutes les 30 s (le fichier fait foi, l'UI ne les modifie pas) :

| Dashboard | Contenu |
|---|---|
| Traitements | Flows Prefect (états, durées), lots d'ingestion et anomalies qualité, journal `tracing.pipeline_runs`, fraîcheur et prix des bougies |
| Modèles | Alias et événements du registre, test final et stratégies simulées du dernier entraînement, historique des runs avec lien vers MLflow |
| Service | Prévisions du `champion` et du `challenger` face à la référence naïve, positions, retours, santé quotidienne (niveau 3), marche / arrêt |
| Charge | Résultats des campagnes Locust, SLO et dimensionnement de l'API |

Les changements du registre et de l'interrupteur apparaissent en annotations sur les courbes.

Les règles d'alerte (`infra/grafana/provisioning/alerting/alertes.yml`, chargées au démarrage de
Grafana, évaluées toutes les 5 min) s'affichent dans Alerting > Alert rules ; aucun canal externe
n'est branché pour l'instant.

| Alerte | Déclenchement |
|---|---|
| Flow Prefect en échec | un flow `FAILED` ou `CRASHED` dans la dernière heure |
| Lot d'ingestion rejeté | un lot rejeté par les contrôles qualité dans la dernière heure |
| Données de marché périmées | dernière bougie de plus de 3 h, système en marche |
| Prévisions absentes | aucune prévision du `champion` depuis 3 h, système en marche |
| Version de modèle dégradée | une version moins bonne que la référence naïve le dernier jour de santé |
| Système arrêté | l'interrupteur est sur arrêt (information) |

## Rejeu 12 mois (`src/replay/`)

Simule des mois d'exploitation sur l'historique, avec le code de production (`training`, `serving`,
`registry`) : chaque mois commence par un entraînement daté du 1er (`as_of`, bougies clôturées avant
uniquement), puis la règle de promotion, puis heure par heure les prévisions du `champion` et du `challenger`,
les retours à horizon clôturé, la santé quotidienne et le retour arrière après 3 jours dégradés.

```powershell
uv run python -m replay --start 2025-11 --months 12 --models volatilite rendement
```

- Registre séparé `<modèle>-replay` (`volatilite-replay`, `rendement-replay`) ; toutes les traces portent
  `mode = 'replay'`. Le champion live et ses tableaux de bord ne sont jamais touchés.
- Reprise : un mois terminé est noté dans `tracing.pipeline_runs` (`replay-<modèle>`), une relance reprend au
  premier mois manquant. `--reset` efface les traces replay (compte administrateur PostgreSQL) ; les modèles
  `*-replay` se suppriment à la main dans MLflow.
- Dans le rejeu, un niveau 2 en échec n'empêche pas la promotion si le niveau 1 passe (démonstration du cycle
  de vie) ; la raison est tracée dans l'événement du registre. Le live garde la règle complète.
- Grafana : dashboard Service, variable « Mode » = `replay` (lien « Rejeu 12 mois ») ; dashboard Modèles,
  modèle `volatilite-replay` ou `rendement-replay`. Runs, annotations et événements sont placés à la date
  simulée (`as_of`), pas à la date d'écriture.

## Application de pilotage (Streamlit)

`src/app/` (http://localhost:8501, service `app` de `compose.yml`) est la console d'exploitation :
elle passe uniquement par l'API (`API_URL`, défaut http://localhost:8000), ne lit aucune base et ne
garde aucun état. La barre latérale porte l'opérateur puis la navigation. Les opérateurs sont des
profils de démonstration (`admin`, `analyste`, `default`) sans contrôle d'accès : le nom choisi est
tracé par l'API avec chaque action, accompagné d'une raison obligatoire.

| Page | Contenu |
|---|---|
| Projet | Sujet (une carte par modèle), schémas des données, des modèles et de l'infrastructure, tableaux de référence ; statique, s'affiche même API arrêtée |
| Service | Santé des dépendances, interrupteur marche / arrêt (`POST /system/state`) et son historique |
| Prévisions | Dernière prévision du `champion` ou du `challenger` par symbole, référence naïve, écart, position |
| Modèles | Alias, santé quotidienne (niveau 3), événements du registre, promotion manuelle et retour arrière (avec confirmation) |

Hors conteneur : `uv run python -m app` (options Streamlit en plus, ex. `--server.port 8502`).
Les schémas sont rendus dans le navigateur par viz.js, copié dans `src/app/static/` et servi par
Streamlit : aucune connexion internet n'est nécessaire.
Grafana reste en lecture seule : l'interrupteur vit ici, Grafana n'affiche que les bascules.

## Dimensionnement (test de charge)

`loadtest/` mesure ce que l'API tient et recommande la plus petite configuration suffisante.
Locust est dans un groupe de dépendances à part (`load`), absent de l'image.

```powershell
uv run --group load python -m loadtest
uv run --group load python -m loadtest --configs 2x1
```

La première commande lance la campagne complète (environ 45 minutes) et redémarre l'API. La seconde
limite le test à une configuration (processus × cœurs).

**Profils d'utilisateurs** (cible) : 200 bots de trading qui relisent `GET /predictions/latest` toutes les 1 à 3 s
et arrivent en rafale au début de chaque heure ; 20 analystes (prévisions, fiche modèle, état) ; 2 opérateurs
(`/health`, `/system/state`).

**Protocole** : pour chaque configuration de l'API (1, 2 ou 4 processus uvicorn × 0,5 ou 1 cœur, via
`API_WORKERS` / `API_CPUS`), trois scénarios : régime normal (42 utilisateurs, 60 s), pic de début d'heure
(222 utilisateurs arrivés en 10 s, montée comprise dans la mesure), rampe de 100 en 100 bots (20 s par palier)
jusqu'au premier palier hors SLO. CPU et mémoire de `api` et `postgres` relevés par `docker stats`.
**SLO** : p95 < 200 ms et moins de 1 % d'erreurs. **Règle** : plus petite configuration (CPU puis processus) qui
tient le pic et le double de la cible (400 bots) sur la rampe. Résultats dans `tracing.load_tests`, le dashboard
Grafana « Charge » et `loadtest/results/<campagne>.json`.

**Résultats** (campagne `2026-11-18T2120Z`, toutes erreurs < 0,02 %) :

| Configuration | p95 au pic | Bots tenus (rampe) | Débit à ce palier | CPU api | Mémoire api |
|---|---|---|---|---|---|
| 1 processus, 0,5 cœur | 71 ms | 200 | 105 req/s | 23 % | 166 Mio |
| 2 processus, 0,5 cœur | 54 ms | 200 | 106 req/s | 24 % | 323 Mio |
| 4 processus, 0,5 cœur | 49 ms | 200 | 106 req/s | 24 % | 573 Mio |
| **1 processus, 1 cœur** | **36 ms** | **500** | **255 req/s** | 51 % | 178 Mio |
| 2 processus, 1 cœur | 23 ms | 600 | 304 req/s | 59 % | 345 Mio |
| 4 processus, 1 cœur | 22 ms | 600 | 303 req/s | 59 % | 599 Mio |

**Recommandation : 1 processus uvicorn, 1 cœur, ~200 Mio de mémoire** (configuration par défaut de `.env.example`).
Elle tient le pic avec un p95 de 36 ms et 500 bots avant rupture, soit 2,5 fois la cible. Avec 0,5 cœur, toutes
les configurations cassent dès 300 bots : la marge de 2× n'est pas tenue. Passer à 2 processus n'apporte que
+100 bots pour deux fois plus de mémoire, et 4 processus rien de plus. PostgreSQL reste sous 10 % de CPU : la base
n'est pas le goulot.

**Limites** :
- client et serveur sur le même PC : Locust (un seul processus) partage les cœurs avec la stack ;
- au point de rupture, le CPU de l'API reste vers 50-60 % et le plafond (~300 req/s) est le même pour 2 et
  4 processus : la limite vient probablement du client Locust ou du pool de connexions PostgreSQL
  (`API_DB_POOL_MAX=5` par processus), pas du calcul de l'API ;
- `/health` et `/models/{nom}` interrogent MLflow (p95 ~400 ms et ~150 ms en régime normal), d'où un p95 global
  de 110 ms sous faible charge : ces routes de supervision sont rares, le SLO vise surtout `/predictions/latest`
  (p95 7 ms) ;
- mesures de 20 à 60 s par palier, une seule campagne.

## Tests

Par défaut, pytest exclut les tests d'intégration, qui nécessitent la stack Docker Compose :

```powershell
uv run pytest
docker compose up -d --build --wait
uv run pytest -m integration
```

Les tests d'intégration créent leurs objets (modèles, traces, dashboards, runs Prefect) avec un
suffixe unique et les suppriment à la fin ; les traces sont effacées avec le compte administrateur
PostgreSQL (`POSTGRES_PASSWORD`), `api_writer` n'ayant pas le droit DELETE. Les runs MLflow
restent dans l'expérience `integration-tests`, séparée de celle du projet.

## Qualité

```powershell
uv run ruff check src tests
uv run ruff format src tests
```
