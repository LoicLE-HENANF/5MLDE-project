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


# -- bougies au format Binance --------------------------------------------------------------
HOUR_MS = 3_600_000
START_MS = 1_735_689_600_000  # 2025-01-01T00:00:00Z


def raw_rows(n=3, start_ms=START_MS, micros=False, close=100.0, step=1.0):
    """Lignes brutes au format Binance (chaînes, comme l'API REST et les CSV)."""
    rows = []
    for i in range(n):
        open_ms = start_ms + i * HOUR_MS
        price = close + i * step
        if micros:
            open_t, close_t = open_ms * 1000, (open_ms + HOUR_MS) * 1000 - 1
        else:
            open_t, close_t = open_ms, open_ms + HOUR_MS - 1
        rows.append(
            [
                str(open_t),
                f'{price:.8f}',
                f'{price + 2:.8f}',
                f'{price - 2:.8f}',
                f'{price + 1:.8f}',
                '10.5',
                str(close_t),
                '1050.0',
                '42',
                '4.0',
                '400.0',
                '0',
            ]
        )
    return rows
