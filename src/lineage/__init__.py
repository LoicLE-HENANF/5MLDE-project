"""Traçabilité : d'où vient exactement un modèle.

- `provenance` : version, commit et empreintes du code et des dépendances qui ont produit un run ou un modèle

Prévu : lien entre le jeu de données (`marketdata.identity`) et le modèle entraîné dessus.
"""

from lineage.errors import ProvenanceError

__all__ = ['ProvenanceError']
