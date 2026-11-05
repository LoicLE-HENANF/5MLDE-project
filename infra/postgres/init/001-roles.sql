-- Initialisation PostgreSQL du projet : rôles, bases et droits (aucune table métier ici).
-- Exécuté automatiquement par l'image postgres au PREMIER démarrage du volume, en superutilisateur.
-- Les tables de traçabilité seront créées par une migration dédiée une fois les données choisies.
\set ON_ERROR_STOP on
\getenv api_password API_DB_PASSWORD
\getenv grafana_password GRAFANA_DB_PASSWORD
\getenv prefect_password PREFECT_DB_PASSWORD

-- Rôles applicatifs (moindre privilège) :
--   api_writer     : l'API écrit ses traces (prédictions, feedbacks, erreurs) dans mlops.tracing
--   grafana_reader : lecture seule pour les dashboards (traces API + runs Prefect)
--   prefect        : propriétaire de la base du serveur Prefect
CREATE ROLE api_writer LOGIN PASSWORD :'api_password';
CREATE ROLE grafana_reader LOGIN PASSWORD :'grafana_password';
CREATE ROLE prefect LOGIN PASSWORD :'prefect_password';
ALTER ROLE grafana_reader SET statement_timeout = '10s';
ALTER ROLE grafana_reader SET default_transaction_read_only = on;

-- Base mlops (courante) : schéma de traçabilité.
REVOKE ALL ON DATABASE mlops FROM PUBLIC;
GRANT CONNECT ON DATABASE mlops TO api_writer, grafana_reader;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
CREATE SCHEMA tracing;
REVOKE ALL ON SCHEMA tracing FROM PUBLIC;
GRANT USAGE ON SCHEMA tracing TO api_writer, grafana_reader;
-- Droits appliqués aux futures tables créées par postgres (migrations) dans tracing.
-- Pas d'UPDATE/DELETE pour l'API : les traces sont en ajout seul.
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA tracing
    GRANT SELECT, INSERT ON TABLES TO api_writer;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA tracing
    GRANT USAGE, SELECT ON SEQUENCES TO api_writer;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA tracing
    GRANT SELECT ON TABLES TO grafana_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA tracing
    GRANT EXECUTE ON FUNCTIONS TO api_writer, grafana_reader;

-- Base prefect : créée vide, le serveur Prefect y applique ses propres migrations.
CREATE DATABASE prefect OWNER prefect;
REVOKE ALL ON DATABASE prefect FROM PUBLIC;
GRANT CONNECT ON DATABASE prefect TO grafana_reader;

\connect prefect
-- Extension utilisée par Prefect pour ses index de recherche ; créée ici en superutilisateur
-- car le rôle prefect n'a pas le droit de la créer.
CREATE EXTENSION IF NOT EXISTS pg_trgm;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT CREATE ON SCHEMA public TO prefect;
-- Grafana lit les tables que Prefect créera (flow_run, task_run, deployment...).
ALTER DEFAULT PRIVILEGES FOR ROLE prefect IN SCHEMA public
    GRANT SELECT ON TABLES TO grafana_reader;
