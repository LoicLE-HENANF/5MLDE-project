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

-- Une ligne par lot ingéré (un symbole, une période), accepté ou rejeté.
-- source_sha256 : archive vérifiée par son .CHECKSUM, ou réponse REST ; data_sha256 : empreinte
-- marketdata.identity des bougies acceptées ; issues : anomalies qualité (bloquantes ou signalées).
CREATE TABLE tracing.ingestion_batches (
    id                  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source              text        NOT NULL CHECK (source IN ('archive', 'rest')),
    symbol              text        NOT NULL CHECK (length(symbol) BETWEEN 1 AND 20),
    period_start        timestamptz NOT NULL,
    period_end          timestamptz NOT NULL,
    status              text        NOT NULL CHECK (status IN ('accepted', 'rejected')),
    rows                integer     NOT NULL CHECK (rows >= 0),
    source_sha256       text        CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
    data_sha256         text        CHECK (data_sha256 ~ '^[0-9a-f]{64}$'),
    issues              jsonb       NOT NULL DEFAULT '[]'::jsonb,
    prefect_flow_run_id uuid,
    recorded_at         timestamptz NOT NULL DEFAULT now(),
    CHECK (period_end > period_start)
);
CREATE INDEX ingestion_batches_symbol_recorded ON tracing.ingestion_batches (symbol, recorded_at DESC);
