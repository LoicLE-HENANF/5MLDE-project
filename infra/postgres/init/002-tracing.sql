-- Tables de traçabilité indépendantes des données (base mlops, schéma tracing).
-- Écrites par connectors.postgres (rôle api_writer, ajout seul), lues par Grafana (grafana_reader) :
-- les droits viennent des ALTER DEFAULT PRIVILEGES de 001-roles.sql.
-- Les colonnes insérables doivent rester alignées sur connectors.postgres.TABLES.
\set ON_ERROR_STOP on

-- Une ligne par exécution terminée d'un pipeline (collecte, entraînement, évaluation...).
CREATE TABLE tracing.pipeline_runs (
    id                  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline            text        NOT NULL CHECK (length(pipeline) BETWEEN 1 AND 100),
    status              text        NOT NULL CHECK (status IN ('success', 'failure', 'cancelled')),
    started_at          timestamptz NOT NULL,
    ended_at            timestamptz NOT NULL,
    prefect_flow_run_id uuid,
    mlflow_run_id       text,
    details             jsonb       NOT NULL DEFAULT '{}'::jsonb,
    recorded_at         timestamptz NOT NULL DEFAULT now(),
    CHECK (ended_at >= started_at)
);
CREATE INDEX pipeline_runs_pipeline_started ON tracing.pipeline_runs (pipeline, started_at DESC);

-- Journal des événements du registre MLflow (enregistrement, promotion d'alias, retrait) :
-- l'historique « quel modèle était en service quand » que les alias MLflow ne gardent pas.
CREATE TABLE tracing.model_events (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name    text        NOT NULL CHECK (length(model_name) BETWEEN 1 AND 100),
    version       integer     NOT NULL CHECK (version > 0),
    event         text        NOT NULL CHECK (event IN ('registered', 'promoted', 'retired')),
    alias         text,
    mlflow_run_id text,
    model_uri     text        NOT NULL,
    details       jsonb       NOT NULL DEFAULT '{}'::jsonb,
    recorded_at   timestamptz NOT NULL DEFAULT now(),
    CHECK (event = 'registered' OR alias IS NOT NULL)
);
CREATE INDEX model_events_model_recorded ON tracing.model_events (model_name, recorded_at DESC);
