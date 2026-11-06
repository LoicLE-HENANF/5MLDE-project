"""Fixtures communes : un environnement complet et valide, et des transports HTTP simulés.

Les tests unitaires ne touchent jamais au réseau ni au .env réel : chaque config est construite
depuis le dictionnaire `env` (copie modifiable par test).
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

VALID_ENV: dict[str, str] = {
    'MLFLOW_TRACKING_URI': 'http://mlflow.test:5000',
    'MLFLOW_EXPERIMENT_NAME': 'exp-test',
    'PREFECT_API_URL': 'http://prefect.test:4200/api',
    'GRAFANA_URL': 'http://grafana.test:3000',
    'GRAFANA_ADMIN_USER': 'admin',
    'GRAFANA_ADMIN_PASSWORD': 'secret-admin',
    'POSTGRES_HOST': 'db.test',
    'POSTGRES_PORT': '5432',
    'API_DB_PASSWORD': 'secret-api',
}


@pytest.fixture
def env() -> dict[str, str]:
    return dict(VALID_ENV)


Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def http_client() -> Callable[..., httpx.Client]:
    """http_client(handler, base_url=None) : client httpx dont les réponses viennent de handler."""
    clients: list[httpx.Client] = []

    def build(handler: Handler, base_url: str = '') -> httpx.Client:
        client = httpx.Client(transport=httpx.MockTransport(handler), base_url=base_url)
        clients.append(client)
        return client

    yield build
    for client in clients:
        client.close()


def refuse(request: httpx.Request) -> httpx.Response:
    """Handler simulant un service éteint (connexion refusée)."""
    raise httpx.ConnectError('Connection refused', request=request)
