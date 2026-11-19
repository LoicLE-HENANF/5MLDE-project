#!/bin/sh
# Volume neuf uniquement (exécuté par l'image postgres avant les *.sql de ce dossier).
# L'image va jouer tous les *.sql : on les inscrit d'avance dans public.schema_migrations
# pour que le service postgres-migrate (infra/postgres/migrate.sh) ne les rejoue pas.
# Si un script échoue, l'initialisation du volume échoue de toute façon (down -v puis up).
# Fichier sourcé par l'entrypoint : pas de `set -e` ni d'`exit` ici.

psql -X -v ON_ERROR_STOP=1 --no-password --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -q \
    -c "CREATE TABLE public.schema_migrations (
            filename   text PRIMARY KEY,
            applied_at timestamptz NOT NULL DEFAULT now()
        )"

for f in /docker-entrypoint-initdb.d/*.sql; do
    psql -X -v ON_ERROR_STOP=1 --no-password --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -q \
        -c "INSERT INTO public.schema_migrations (filename) VALUES ('$(basename "$f")')"
done
