# Image du projet : worker Prefect (flows planifiés) et API FastAPI (compose.yml).
#
# Le paquet est installé en copie figée (non éditable) avec les dépendances exactes de uv.lock ;
# sa version vient de Git au build (tag exact -> 0.2.0, sinon 0.2.1.devN+g<sha>[.d<date>]).
# Seule une image construite sur un tag propre peut publier des modèles au registre.
#   docker compose build worker api

FROM ghcr.io/astral-sh/uv:0.9.26-python3.12-bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
# Arbre de travail Windows (CRLF) monté sous Linux : sans ces réglages Git le verrait modifié.
RUN git config --global --add safe.directory /src && git config --global core.autocrlf true
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv UV_PYTHON_DOWNLOADS=never
# Montage du contexte (lecture-écriture, jeté après l'étape) : .git n'entre dans aucune couche.
RUN --mount=type=bind,source=.,target=/src,rw \
    --mount=type=cache,target=/root/.cache/uv \
    cd /src && git update-index -q --refresh || true \
    && cd /src && uv sync --frozen --no-dev --no-editable \
    && /app/.venv/bin/python -c "import importlib.metadata as m; print('version', m.version('5MLDE-project'))"

FROM python:3.12.12-slim-bookworm
RUN useradd --create-home --uid 1000 app && mkdir -p /data && chown app /data
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 DATA_DIR=/data
USER app
WORKDIR /home/app
CMD ["python", "-m", "flows.serve"]
