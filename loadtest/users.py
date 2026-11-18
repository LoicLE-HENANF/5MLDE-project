"""Profils d'utilisateurs Locust de l'API (voir loadtest.plan pour les effectifs)."""

from __future__ import annotations

import random

from locust import HttpUser, between, task

MODELS = ('volatilite', 'rendement')
TIMEOUT_S = 10


class Bot(HttpUser):
    """Bot de trading : relit la dernière prévision en boucle jusqu'à la nouvelle bougie."""

    abstract = True
    wait_time = between(1, 3)

    @task
    def latest(self) -> None:
        self.client.get(
            '/predictions/latest',
            params={'model': random.choice(MODELS)},
            name='/predictions/latest',
            timeout=TIMEOUT_S,
        )


class Analyst(HttpUser):
    """Analyste (app ou API) : prévisions, puis fiche d'un modèle (registre MLflow + santé)."""

    abstract = True
    wait_time = between(2, 5)

    @task(3)
    def latest(self) -> None:
        self.client.get(
            '/predictions/latest',
            params={'model': random.choice(MODELS), 'alias': random.choice(('champion', 'challenger'))},
            name='/predictions/latest',
            timeout=TIMEOUT_S,
        )

    @task(1)
    def model(self) -> None:
        name = random.choice(MODELS)
        self.client.get(f'/models/{name}', params={'days': 7}, name='/models/[name]', timeout=TIMEOUT_S)

    @task(1)
    def system(self) -> None:
        self.client.get('/system/state', name='/system/state', timeout=TIMEOUT_S)


class Operator(HttpUser):
    """Opérateur : supervision de la santé et de l'interrupteur marche / arrêt."""

    abstract = True
    wait_time = between(5, 10)

    @task(2)
    def health(self) -> None:
        self.client.get('/health', name='/health', timeout=TIMEOUT_S)

    @task(1)
    def system(self) -> None:
        self.client.get('/system/state', params={'history': 5}, name='/system/state', timeout=TIMEOUT_S)


def sized(base: type[HttpUser], count: int) -> type[HttpUser]:
    """Sous-classe concrète lancée exactement `count` fois (fixed_count Locust)."""
    return type(base.__name__, (base,), {'abstract': False, 'fixed_count': count, 'weight': 1})
