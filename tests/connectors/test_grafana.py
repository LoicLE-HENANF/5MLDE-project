"""Connecteur Grafana : API HTTP simulée (httpx.MockTransport)."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime

import httpx
import pytest

from connectors.config import GrafanaConfig
from connectors.errors import ConnectorError
from connectors.grafana import GrafanaConnector
from tests.conftest import refuse


@pytest.fixture
def config(env) -> GrafanaConfig:
    return GrafanaConfig.from_env(env)


def healthy(request: httpx.Request) -> httpx.Response:
    if request.url.path == '/api/health':
        return httpx.Response(200, json={'database': 'ok', 'version': '13.2.3'})
    if request.url.path.endswith('/health'):
        return httpx.Response(200, json={'status': 'OK', 'message': 'Database Connection OK'})
    return httpx.Response(404)


def connector_with(config, http_client, handler) -> GrafanaConnector:
    return GrafanaConnector(config, client=http_client(handler, base_url=config.url))


# -- authentification --------------------------------------------------------------------------
def _auth_header(config) -> str:
    seen = {}

    def capture(request):
        seen['auth'] = request.headers.get('authorization', '')
        return httpx.Response(200, json={'database': 'ok'})

    with GrafanaConnector(config) as grafana:
        grafana.client._transport = httpx.MockTransport(capture)
        grafana._request('GET', '/api/health')
    return seen['auth']


def test_basic_auth_by_default(config):
    expected = base64.b64encode(b'admin:secret-admin').decode()
    assert _auth_header(config) == f'Basic {expected}'


def test_token_takes_precedence(env):
    env['GRAFANA_TOKEN'] = 'glsa_x'
    assert _auth_header(GrafanaConfig.from_env(env)) == 'Bearer glsa_x'


# -- santé -------------------------------------------------------------------------------------
def test_check_ok(config, http_client):
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return healthy(request)

    connector_with(config, http_client, handler).check()
    assert seen == [
        '/api/health',
        '/api/datasources/uid/projet-traces/health',
        '/api/datasources/uid/projet-prefect/health',
        '/api/datasources/uid/projet-mlflow/health',
    ]


def test_check_server_down(config, http_client):
    with pytest.raises(ConnectorError, match=r'injoignable.*GRAFANA_URL'):
        connector_with(config, http_client, refuse).check()


def test_check_bad_credentials(config, http_client):
    with pytest.raises(ConnectorError, match='identifiants refusés'):
        connector_with(config, http_client, lambda r: httpx.Response(401)).check()


def test_check_internal_db_down(config, http_client):
    def handler(request):
        return httpx.Response(200, json={'database': 'failing'})

    with pytest.raises(ConnectorError, match='base interne KO'):
        connector_with(config, http_client, handler).check()


def test_check_datasource_missing(config, http_client):
    def handler(request):
        if 'projet-mlflow' in request.url.path:
            return httpx.Response(404, json={'message': 'Data source not found'})
        return healthy(request)

    with pytest.raises(ConnectorError, match="datasource 'projet-mlflow' absente"):
        connector_with(config, http_client, handler).check()


def test_check_datasource_cannot_reach_postgres(config, http_client):
    def handler(request):
        if 'projet-traces' in request.url.path:
            return httpx.Response(400, json={'status': 'ERROR', 'message': 'password authentication failed'})
        return healthy(request)

    with pytest.raises(ConnectorError, match=r'password authentication failed.*GRAFANA_DB_PASSWORD'):
        connector_with(config, http_client, handler).check()


# -- écritures ---------------------------------------------------------------------------------
def test_upsert_dashboard(config, http_client):
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={'uid': 'd1', 'url': '/d/d1/suivi', 'status': 'success'})

    url = connector_with(config, http_client, handler).upsert_dashboard(
        {'uid': 'd1', 'title': 'Suivi', 'id': 99, 'panels': []}, folder_uid='f1', message='maj'
    )
    assert url == 'http://grafana.test:3000/d/d1/suivi'
    assert sent['dashboard']['id'] is None and sent['overwrite'] is True
    assert sent['folderUid'] == 'f1' and sent['message'] == 'maj'


def test_upsert_dashboard_requires_uid(config, http_client):
    with pytest.raises(ValueError):
        connector_with(config, http_client, healthy).upsert_dashboard({'title': 'x'})


def test_upsert_dashboard_server_error(config, http_client):
    def handler(request):
        return httpx.Response(412, text='version-mismatch')

    with pytest.raises(ConnectorError, match='HTTP 412'):
        connector_with(config, http_client, handler).upsert_dashboard({'uid': 'd', 'title': 't'})


def test_annotate(config, http_client):
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={'id': 5, 'message': 'Annotation added'})

    when = datetime(2026, 1, 1, tzinfo=UTC)
    assert (
        connector_with(config, http_client, handler).annotate(
            'promotion v3', tags=['modele'], when=when, dashboard_uid='d1'
        )
        == 5
    )
    assert sent == {'text': 'promotion v3', 'tags': ['modele'], 'time': 1767225600000, 'dashboardUID': 'd1'}


def test_annotate_rejects_naive_datetime(config, http_client):
    with pytest.raises(ValueError, match='fuseau'):
        connector_with(config, http_client, healthy).annotate('x', when=datetime(2026, 1, 1))
