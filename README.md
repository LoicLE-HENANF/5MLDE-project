# 5MLDE-project

Mini projet du module 5MLDE — énoncé : [subject.md](subject.md).

## Prérequis

- Docker (Compose v2)
- [uv](https://docs.astral.sh/uv/) — Python 3.12 est installé automatiquement
- un clone git du dépôt : la version du paquet est calculée depuis les tags git

## Stack locale (Docker Compose)

```powershell
copy .env.example .env        # puis adapter si besoin (ports, mots de passe, adresses)
docker compose up -d --build --wait   # démarre et attend que tous les services soient healthy
docker compose down           # arrêt (ajouter -v pour repartir de volumes vides)
```

| Service | URL (hôte) | Rôle |
|---|---|---|
| MLflow 3.16.1 | http://localhost:5000 | Expériences, registre de modèles, artefacts (backend PostgreSQL, base `mlflow`) |
| Prefect 3.8.7 | http://localhost:4200 | Orchestration des flows (backend PostgreSQL, base `prefect`) |
| PostgreSQL 18.6 | `localhost:5432` | Bases `mlops` (schéma `tracing`), `prefect`, `mlflow` |
| Grafana 13.2.3 | http://localhost:3000 | Supervision (admin / `5mlde-project-admin` par défaut) |

Tous les ports sont liés à `127.0.0.1` uniquement et se changent dans `.env`.
Les mots de passe PostgreSQL ne sont appliqués qu'à la création du volume `postgres_data`.

### Rôles PostgreSQL

Créés par `infra/postgres/init/001-roles.sql` au premier démarrage du volume ; les tables de
traçabilité sont créées par `infra/postgres/init/002-tracing.sql`.

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
uv run python -m connectors.health              # toutes les briques
uv run python -m connectors.health mlflow postgres
```

### Traçabilité des modèles

Chaque run et chaque modèle MLflow porte la provenance du code qui l'a produit (tags `code.*`) :
version calculée depuis git (`0.2.0` = tag `v0.2.0`, `0.2.1.devN+g<sha>` = commit, suffixe
`.d<date>` = arbre modifié), empreinte SHA-256 des paquets du projet (`connectors`, `features`, `flows`,
`ingestion`, `lineage` et `marketdata`) et des dépendances installées. Le code des paquets et la liste
exacte des dépendances sont embarqués dans le modèle.

`register_model` refuse un modèle dont le code n'est pas identifiable de façon fiable : arbre
git modifié ou installation éditable. Pour produire un modèle publiable :

```powershell
git tag v0.2.0                                  # ou committer : version 0.2.1.devN+g<sha>
uv run --no-editable python -m <votre_module>   # copie figée du code, version exacte
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

## Paquet `training`

`training.splits.walk_forward_splits` produit des indices positionnels pour des folds expansifs :
12 mois initiaux, validations mensuelles, embargo au moins égal à l'horizon et trois derniers mois
réservés au test final. Les lignes sans cible sont exclues si `target_column` est fourni.

## Tests

```powershell
uv run pytest                    # tests unitaires : services simulés, aucune stack nécessaire
docker compose up -d --build --wait
uv run pytest -m integration     # tests d'intégration contre la stack réelle
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
