"""Erreurs de traçabilité."""

from __future__ import annotations


class ProvenanceError(ValueError):
    """Le code qui a produit un modèle n'est pas identifiable de façon fiable (non publiable)."""
