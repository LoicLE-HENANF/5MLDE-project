# Image MLflow officielle + pilote PostgreSQL (absent de l'image) pour le backend store.
# Installé au build (et non au démarrage) : le conteneur démarre sans accès réseau.
FROM ghcr.io/mlflow/mlflow:v3.16.1
RUN pip install --no-cache-dir "psycopg2-binary>=2.9.10,<3"
