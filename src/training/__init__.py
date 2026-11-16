"""Outils de découpage temporel des données d'entraînement."""

from training.splits import holdout_split, walk_forward_splits

__all__ = ['holdout_split', 'walk_forward_splits']
