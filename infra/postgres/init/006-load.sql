-- Résultats des tests de charge de l'API (loadtest/, base mlops, schéma tracing), lus par Grafana.
-- Une ligne par campagne × configuration (workers uvicorn, limite CPU) × scénario ; ajout seul.
-- Les colonnes insérables doivent rester alignées sur connectors.postgres.TABLES.
--
-- Sur un volume déjà initialisé, appliquer à la main :
--   docker compose exec -T postgres psql -U postgres -d mlops -f /docker-entrypoint-initdb.d/006-load.sql
\set ON_ERROR_STOP on

CREATE TABLE tracing.load_tests (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    campaign     text             NOT NULL CHECK (length(campaign) BETWEEN 1 AND 100),
    scenario     text             NOT NULL CHECK (scenario IN ('normal', 'pic', 'rupture')),
    workers      integer          NOT NULL CHECK (workers > 0),
    cpus         double precision NOT NULL CHECK (cpus > 0),
    -- utilisateurs simulés (rupture : dernier palier tenant le SLO, 0 si aucun)
    users        integer          NOT NULL CHECK (users >= 0),
    duration_s   double precision NOT NULL CHECK (duration_s > 0),
    requests     bigint           NOT NULL CHECK (requests >= 0),
    failures     bigint           NOT NULL CHECK (failures >= 0 AND failures <= requests),
    rps          double precision NOT NULL CHECK (rps >= 0),
    p50_ms       double precision,
    p95_ms       double precision,
    p99_ms       double precision,
    max_ms       double precision,
    -- docker stats pendant la mesure (100 % = un cœur)
    api_cpu_pct  double precision,
    api_mem_mb   double precision,
    pg_cpu_pct   double precision,
    pg_mem_mb    double precision,
    slo_ok       boolean          NOT NULL,
    recommended  boolean          NOT NULL DEFAULT false,
    details      jsonb            NOT NULL DEFAULT '{}'::jsonb,
    recorded_at  timestamptz      NOT NULL DEFAULT now()
);
CREATE INDEX load_tests_campaign ON tracing.load_tests (campaign, recorded_at DESC);
