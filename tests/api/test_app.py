"""Tests de l'API FastAPI : connecteurs remplacés par des doubles, registre simulé."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api.app as api_app
from api.app import Services, create_app
from connectors.errors import ConnectorError
from registry import RegistryError

STATE = {'active': True, 'changed_by': None, 'reason': None, 'recorded_at': None}
ALIASES = {'champion': '2', 'challenger': None, 'precedent': None}


@pytest.fixture
def services(monkeypatch) -> Services:
    db = MagicMock()
    db.system_state.return_value = dict(STATE)
    db.select.return_value = []
    db.record_system_change.return_value = 7
    monkeypatch.setattr(api_app, 'status', lambda mlflow, name: dict(ALIASES))
    return Services(db=db, mlflow=MagicMock())


@pytest.fixture
def client(services):
    with TestClient(create_app(lambda: services)) as test_client:
        yield test_client


# -- santé ----------------------------------------------------------------------


def test_health_ok_reports_system_and_aliases(client):
    response = client.get('/health')
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ok'
    assert body['services'] == {'postgres': 'ok', 'mlflow': 'ok'}
    assert body['system']['active'] is True
    assert body['models']['volatilite'] == ALIASES


def test_health_is_degraded_when_a_dependency_is_down(client, services):
    services.mlflow.check.side_effect = ConnectorError('MLflow', 'x', 'injoignable')
    response = client.get('/health')
    assert response.status_code == 503
    body = response.json()
    assert body['status'] == 'degraded'
    assert body['services']['mlflow'].startswith('erreur')
    assert 'models' not in body


def test_db_is_closed_on_shutdown(services):
    with TestClient(create_app(lambda: services)):
        pass
    services.db.close.assert_called_once()


# -- lectures -------------------------------------------------------------------


def test_latest_predictions_reads_the_requested_alias(client, services):
    services.db.select.return_value = [{'symbol': 'BTCUSDT', 'prediction': 0.02}]
    response = client.get('/predictions/latest', params={'alias': 'challenger'})
    assert response.status_code == 200
    assert response.json()['predictions'] == [{'symbol': 'BTCUSDT', 'prediction': 0.02}]
    assert services.db.select.call_args.args[1] == ('volatilite', 'challenger')


@pytest.mark.parametrize(
    ('path', 'params'),
    [('/predictions/latest', {'model': 'inconnu'}), ('/models/inconnu', {})],
)
def test_unknown_model_is_404(client, path, params):
    assert client.get(path, params=params).status_code == 404


def test_invalid_alias_is_422(client):
    assert client.get('/predictions/latest', params={'alias': 'precedent'}).status_code == 422


def test_model_returns_aliases_health_and_events(client, services):
    response = client.get('/models/volatilite', params={'days': 3})
    assert response.status_code == 200
    body = response.json()
    assert body['aliases'] == ALIASES
    query, params = services.db.select.call_args_list[0].args
    assert params == ('volatilite', 3)
    # santé du service réel uniquement, sur `days` jours (pas `days` lignes)
    text = query.as_string(None)
    assert "mode = 'live'" in text and 'current_date - %s' in text and 'LIMIT' not in text


# -- actions sur le registre ----------------------------------------------------


def test_promote_passes_version_and_traced_reason(client, services, monkeypatch):
    calls = []

    def fake_promote(mlflow, name, version, *, reason, record):
        calls.append((name, version, reason, record))
        return SimpleNamespace(version=version)

    monkeypatch.setattr(api_app, 'promote', fake_promote)
    response = client.post(
        '/models/volatilite/promote',
        json={'version': 3, 'reason': 'meilleur QLIKE', 'changed_by': 'loic'},
    )
    assert response.status_code == 200
    assert response.json()['champion'] == '3'
    assert calls == [('volatilite', 3, 'meilleur QLIKE (par loic)', services.db.record_model_event)]


def test_promote_requires_a_reason(client):
    response = client.post(
        '/models/volatilite/promote', json={'version': 3, 'reason': '', 'changed_by': 'loic'}
    )
    assert response.status_code == 422


@pytest.mark.parametrize('action', ['promote', 'rollback'])
def test_registry_errors_are_409(client, monkeypatch, action):
    def refuse(*args, **kwargs):
        raise RegistryError('refusé')

    monkeypatch.setattr(api_app, action, refuse)
    body = {'version': 1, 'reason': 'r', 'changed_by': 'loic'} if action == 'promote' else None
    response = client.post(f'/models/volatilite/{action}', json=body)
    assert response.status_code == 409
    assert response.json()['detail'] == 'refusé'


def test_rollback_returns_restored_version(client, monkeypatch):
    monkeypatch.setattr(api_app, 'rollback', lambda *a, **k: SimpleNamespace(version=1))
    response = client.post('/models/volatilite/rollback')
    assert response.status_code == 200
    assert response.json()['champion'] == '1'


# -- interrupteur ---------------------------------------------------------------


def test_system_state_with_history(client, services):
    services.db.select.return_value = [{'active': False}]
    body = client.get('/system/state', params={'history': 5}).json()
    assert body['state'] == STATE
    assert body['history'] == [{'active': False}]


def test_system_state_without_history_skips_the_query(client, services):
    assert client.get('/system/state', params={'history': 0}).json()['history'] == []
    services.db.select.assert_not_called()


def test_change_system_state_records_who_and_why(client, services):
    response = client.post(
        '/system/state', json={'active': False, 'changed_by': 'loic', 'reason': 'maintenance'}
    )
    assert response.status_code == 200
    assert response.json()['id'] == 7
    services.db.record_system_change.assert_called_once_with(
        active=False, changed_by='loic', reason='maintenance'
    )


@pytest.mark.parametrize(
    'body',
    [
        {'active': False, 'changed_by': '', 'reason': 'x'},
        {'active': False, 'changed_by': 'loic', 'reason': ''},
        {'changed_by': 'loic', 'reason': 'x'},
    ],
)
def test_change_system_state_validates_input(client, body):
    assert client.post('/system/state', json=body).status_code == 422
