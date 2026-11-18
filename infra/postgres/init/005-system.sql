-- État marche / arrêt du système (base mlops, schéma tracing), modifié uniquement via l'API.
-- Ajout seul, comme les autres traces : l'état courant est la dernière ligne, l'historique des
-- changements (qui, quand, pourquoi, ancienne et nouvelle valeur) se lit dans la même table.
-- Les colonnes insérables doivent rester alignées sur connectors.postgres.TABLES.
--
-- Sur un volume déjà initialisé, appliquer à la main :
--   docker compose exec -T postgres psql -U postgres -d mlops -f /docker-entrypoint-initdb.d/005-system.sql
\set ON_ERROR_STOP on

CREATE TABLE tracing.system_changes (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    active      boolean     NOT NULL,
    previous    boolean,
    changed_by  text        NOT NULL CHECK (length(changed_by) BETWEEN 1 AND 100),
    reason      text        NOT NULL CHECK (length(reason) BETWEEN 1 AND 500),
    recorded_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX system_changes_recorded ON tracing.system_changes (recorded_at DESC);
