# 5MLDE-project

Mini projet du module 5MLDE — énoncé : [subject.md](subject.md).

## Stack locale (Docker Compose)

```powershell
copy .env.example .env        # facultatif : ports et mots de passe
docker compose up -d --wait   # démarre et attend que tous les services soient healthy
docker compose down           # arrêt (ajouter -v pour repartir de volumes vides)
```

| Service | URL (hôte) | Rôle |
|---|---|---|
| MLflow 3.16.1 | http://localhost:5050 | Suivi des expériences, registre de modèles, artefacts (backend SQLite) |
| Prefect 3.8.7 | http://localhost:4250 | Orchestration des flows (backend PostgreSQL, base `prefect`) |
| PostgreSQL 18.6 | `localhost:5433` | Base `mlops` (schéma `tracing` : traces de l'API) + base `prefect` |
| Grafana 13.2.3 | http://localhost:3030 | Supervision (admin / `5mlde-projet-admin` par défaut) |

Les ports sont décalés par rapport à la stack des TPs (5000/4200/5432/3000) pour que les deux
puissent tourner en même temps sans qu'un client mal configuré écrive dans la mauvaise base.

### Rôles PostgreSQL

Créés par `infra/postgres/init/001-roles.sql` au premier démarrage du volume :

| Rôle | Accès |
|---|---|
| `api_writer` | `mlops.tracing` : SELECT/INSERT (traces en ajout seul) |
| `grafana_reader` | lecture seule (`default_transaction_read_only`, timeout 10 s) sur `mlops.tracing` et `prefect` |
| `prefect` | propriétaire de la base `prefect` (migrations gérées par le serveur Prefect) |

Grafana est provisionné automatiquement : deux datasources (`projet-traces`, `projet-prefect`)
et les dashboards JSON placés dans `infra/grafana/dashboards/`.
