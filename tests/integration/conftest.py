"""Fixtures partagées des tests d'intégration : configuration .env et accès Grafana."""

from __future__ import annotations

import pytest

from connectors.config import GrafanaConfig, load_env
from connectors.grafana import GrafanaConnector


@pytest.fixture(scope='module')
def env() -> dict[str, str]:
    return load_env()


@pytest.fixture(scope='module')
def grafana(env):
    with GrafanaConnector(GrafanaConfig.from_env(env)) as connector:
        yield connector


def grafana_sql(grafana: GrafanaConnector, datasource_uid: str, sql: str) -> list[list]:
    """Requête SQL exécutée PAR Grafana (rôle grafana_reader) ; renvoie les colonnes."""
    response = grafana._request(
        'POST',
        '/api/ds/query',
        json={
            'queries': [
                {
                    'refId': 'A',
                    'datasource': {'uid': datasource_uid},
                    'rawSql': sql,
                    'format': 'table',
                }
            ]
        },
    )
    frames = response.json()['results']['A']['frames']
    return frames[0]['data']['values'] if frames else []
