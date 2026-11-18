"""Registre des modèles : décision de promotion, alias champion/challenger, retour arrière."""

from registry.promotion import (
    CHALLENGER,
    CHAMPION,
    PREVIOUS,
    Decision,
    RegistryError,
    decide,
    model_uri,
    promote,
    publish,
    rollback,
    status,
)

__all__ = [
    'CHALLENGER',
    'CHAMPION',
    'PREVIOUS',
    'Decision',
    'RegistryError',
    'decide',
    'model_uri',
    'promote',
    'publish',
    'rollback',
    'status',
]
