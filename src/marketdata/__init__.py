"""Définition des données de marché : bougies horaires Binance spot.

- `universe` : symboles, intervalle et début de l'historique
- `sources` : format brut des deux sources (archives mensuelles, API REST) et normalisation
- `schema` : schéma pandera d'une bougie normalisée
- `quality` : règles qualité (bloquantes ou signalées)
- `identity` : identité d'un jeu de données (symboles, période, empreinte SHA-256)

Aucune I/O réseau ici : le téléchargement appartient à l'ingestion.
"""
