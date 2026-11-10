"""Ingestion des bougies Binance : téléchargement, contrôles qualité, stockage, traçabilité.

- `download` : client HTTP des deux sources (archives mensuelles vérifiées par `.CHECKSUM`, API REST)
- `store` : historique Parquet (`data/candles/...`) et bougies récentes PostgreSQL (`market.candles`)
- `pipeline` : traitement d'un lot (normalisation, qualité, écriture, trace dans `tracing.ingestion_batches`)

Le format des données est défini dans `marketdata` ; ce paquet ne fait que les déplacer.
"""
