-- Tables du service (base mlops, schéma tracing) : prédictions, retours et santé des modèles.
-- Mêmes règles que 002-tracing.sql : écrites par connectors.postgres (api_writer, ajout seul),
-- lues par Grafana. Les colonnes insérables doivent rester alignées sur connectors.postgres.TABLES.
--
-- Sur un volume déjà initialisé (les scripts d'init ne tournent qu'une fois), appliquer à la main :
--   docker compose exec -T postgres psql -U postgres -d mlops -f /docker-entrypoint-initdb.d/004-serving.sql
\set ON_ERROR_STOP on

-- Une prédiction par (modèle, version, alias, symbole, bougie, mode) : rejouer l'inférence sur la même
-- bougie ne crée pas de doublon (INSERT ... ON CONFLICT DO NOTHING). open_time est l'ouverture de la
-- dernière bougie clôturée au moment de la prédiction ; la cible couvre les horizon_hours suivantes.
CREATE TABLE tracing.predictions (
    id                  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name          text             NOT NULL CHECK (length(model_name) BETWEEN 1 AND 100),
    version             integer          NOT NULL CHECK (version > 0),
    alias               text             NOT NULL CHECK (alias IN ('champion', 'challenger')),
    symbol              text             NOT NULL CHECK (length(symbol) BETWEEN 1 AND 20),
    open_time           timestamptz      NOT NULL,
    horizon_hours       integer          NOT NULL CHECK (horizon_hours > 0),
    prediction          double precision NOT NULL,
    -- prédiction de la référence naïve (persistance, momentum) pour le niveau 3
    baseline            double precision,
    -- poids simulé (paper trading) : champion uniquement, NULL pour le challenger
    position            double precision,
    mode                text             NOT NULL CHECK (mode IN ('live', 'replay')),
    mlflow_run_id       text,
    prefect_flow_run_id uuid,
    recorded_at         timestamptz      NOT NULL DEFAULT now(),
    CHECK (alias = 'champion' OR position IS NULL),
    UNIQUE (model_name, version, alias, symbol, open_time, mode)
);
CREATE INDEX predictions_open_time ON tracing.predictions (model_name, mode, open_time DESC);

-- Valeur réalisée d'une prédiction, écrite une seule fois quand l'horizon est écoulé.
CREATE TABLE tracing.feedback (
    id                  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prediction_id       bigint           NOT NULL UNIQUE REFERENCES tracing.predictions (id),
    realized            double precision NOT NULL,
    -- perte de la prédiction et de la référence sur cette ligne (QLIKE pour la volatilité)
    loss                double precision,
    baseline_loss       double precision,
    prefect_flow_run_id uuid,
    recorded_at         timestamptz      NOT NULL DEFAULT now()
);

-- Niveau 3 : santé quotidienne d'une version en service, calculée sur une fenêtre glissante de retours.
-- degraded = la version fait moins bien que sa référence naïve (métrique) ou que la stratégie brute (PnL).
CREATE TABLE tracing.model_health (
    id                  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name          text             NOT NULL CHECK (length(model_name) BETWEEN 1 AND 100),
    version             integer          NOT NULL CHECK (version > 0),
    mode                text             NOT NULL CHECK (mode IN ('live', 'replay')),
    day                 date             NOT NULL,
    window_hours        integer          NOT NULL CHECK (window_hours > 0),
    samples             integer          NOT NULL CHECK (samples >= 0),
    metric              text             NOT NULL CHECK (length(metric) BETWEEN 1 AND 50),
    value               double precision,
    baseline_value      double precision,
    pnl                 double precision,
    baseline_pnl        double precision,
    degraded            boolean          NOT NULL,
    details             jsonb            NOT NULL DEFAULT '{}'::jsonb,
    prefect_flow_run_id uuid,
    recorded_at         timestamptz      NOT NULL DEFAULT now(),
    UNIQUE (model_name, version, mode, day)
);
CREATE INDEX model_health_day ON tracing.model_health (model_name, mode, day DESC);
