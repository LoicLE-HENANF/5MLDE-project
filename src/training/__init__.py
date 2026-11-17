"""Outils de découpage temporel des données d'entraînement."""

from training.metrics import icir, information_coefficient
from training.splits import holdout_split, walk_forward_splits

__all__ = ['holdout_split', 'icir', 'information_coefficient', 'walk_forward_splits']
