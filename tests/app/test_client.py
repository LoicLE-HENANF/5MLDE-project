"""Client HTTP de l'application : transport simulé, aucun serveur."""

from __future__ import annotations

import json

import httpx
import pytest

from app.client import ApiClient, ApiError, api_url


def _client(handler) -> ApiClient:
    return ApiClient(
        'http://api.test',
        client=httpx.Client(base_url='http://api.test', transport=httpx.MockTransport(handler)),
    )


@pytest.mark.parametrize(
    ('env', 'expected'),
    [
        ({}, 'http://localhost:8000'),
        ({'API_URL': ' http://api:8000/ '}, 'http://api:8000'),
    ],
)
def test_api_url(env, expected):
    assert api_url(env) == expected


@pytest.mark.parametrize('url', ['api:8000', 'ftp://api', 'http://'])
def test_api_url_rejects_invalid(url):
    with pytest.raises(ValueError, match='API_URL'):
        api_url({'API_URL': url})


def test_health_accepts_503():
    client = _client(lambda request: httpx.Response(503, json={'status': 'degraded'}))
    assert client.health() == {'status': 'degraded'}


def test_predictions_sends_model_and_alias():
    seen = []

    def handler(request):
        seen.append(request.url)
        return httpx.Response(200, json={'predictions': [{'symbol': 'BTCUSDT'}]})

    assert _client(handler).predictions('volatilite', 'challenger') == [{'symbol': 'BTCUSDT'}]
    assert seen[0].path == '/predictions/latest'
    assert dict(seen[0].params) == {'model': 'volatilite', 'alias': 'challenger'}


def test_actions_post_payloads():
    bodies = {}

    def handler(request):
        bodies[request.url.path] = json.loads(request.content or b'null')
        return httpx.Response(200, json={'ok': True})

    client = _client(handler)
    client.set_system(active=False, changed_by='loic', reason='maintenance')
    client.promote('volatilite', version=2, changed_by='loic', reason='meilleur')
    client.rollback('volatilite')
    assert bodies['/system/state'] == {'active': False, 'changed_by': 'loic', 'reason': 'maintenance'}
    assert bodies['/models/volatilite/promote'] == {'version': 2, 'changed_by': 'loic', 'reason': 'meilleur'}
    assert bodies['/models/volatilite/rollback'] is None


def test_refusal_carries_api_detail():
    client = _client(lambda request: httpx.Response(409, json={'detail': 'aucune version précédente'}))
    with pytest.raises(ApiError, match='aucune version précédente') as info:
        client.rollback('volatilite')
    assert info.value.status_code == 409


def test_validation_error_is_readable():
    detail = [{'loc': ['body', 'reason'], 'msg': 'String should have at least 1 character'}]
    client = _client(lambda request: httpx.Response(422, json={'detail': detail}))
    with pytest.raises(ApiError, match='reason : String should have at least 1 character'):
        client.set_system(active=True, changed_by='loic', reason='')


def test_unreachable_api():
    def handler(request):
        raise httpx.ConnectError('refused', request=request)

    with pytest.raises(ApiError, match='injoignable'):
        _client(handler).health()
