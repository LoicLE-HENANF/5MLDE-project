#!/bin/sh
# Service compose postgres-migrate : à chaque `docker compose up`, applique les scripts
# infra/postgres/init/*.sql pas encore passés sur ce volume (suivi : public.schema_migrations).
# Un volume neuf est déjà à jour (000-migrations.sh) ; un volume existant reçoit les nouveaux scripts.
# Règle : ne jamais modifier un script déjà publié, en ajouter un nouveau (007-..., 008-...).
set -eu

DIR=/docker-entrypoint-initdb.d
q() { psql -X -v ON_ERROR_STOP=1 -qAt "$@"; }

if [ -z "$(q -c "SELECT to_regclass('public.schema_migrations')")" ]; then
    # Volumes créés avant ce suivi : 001 à 006 y sont déjà (seule config connue en service).
    echo "schema_migrations absente : volume antérieur au suivi, 001 à 006 marqués comme appliqués"
    q <<'SQL'
CREATE TABLE public.schema_migrations (
    filename   text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);
INSERT INTO public.schema_migrations (filename) VALUES
    ('001-roles.sql'), ('002-tracing.sql'), ('003-market.sql'),
    ('004-serving.sql'), ('005-system.sql'), ('006-load.sql');
SQL
fi

for f in "$DIR"/*.sql; do
    name=$(basename "$f")
    if [ -z "$(q -c "SELECT 1 FROM public.schema_migrations WHERE filename = '$name'")" ]; then
        echo "application de $name"
        q -f "$f"
        q -c "INSERT INTO public.schema_migrations (filename) VALUES ('$name')"
    fi
done
echo "migrations à jour"
