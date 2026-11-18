"""Page Streamlit exécutée avec `AppTest`, l'API remplacée par un faux client (aucun réseau)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import client as api

PAGE = str(Path(api.__file__).with_name('main.py'))
NOW = '2026-11-18T10:00:00Z'


class FakeClient:
    base_url = 'http://api.test'

    def __init__(self, *, health_error: bool = False, refuse: bool = False) -> None:
        self.health_error = health_error
        self.refuse = refuse
        self.calls: list[tuple[str, dict]] = []
        self.active = True

    def close(self) -> None:
        pass

    def health(self):
        if self.health_error:
            raise api.ApiError('API injoignable (http://api.test) : ConnectError')
        return {
            'status': 'ok',
            'services': {'postgres': 'ok', 'mlflow': 'ok'},
            'system': {'active': self.active},
        }

    def system(self, history=20):
        return {
            'state': {'active': self.active, 'changed_by': None, 'reason': None, 'recorded_at': None},
            'history': [],
        }

    def predictions(self, model, alias='champion'):
        if model != 'volatilite':
            return []
        return [
            {
                'symbol': 'BTCUSDT',
                'open_time': NOW,
                'horizon_hours': 24,
                'prediction': 0.03,
                'baseline': 0.02,
                'position': 0.5,
                'version': 2,
            }
        ]

    def model(self, name, days=14):
        return {
            'aliases': {'champion': '2', 'challenger': '3', 'precedent': '1'},
            'health': [
                {
                    'day': '2026-11-17',
                    'version': 2,
                    'mode': 'live',
                    'samples': 5,
                    'metric': 'qlike',
                    'value': 0.4,
                    'baseline_value': 0.5,
                    'pnl': 0.01,
                    'baseline_pnl': 0.0,
                    'degraded': False,
                }
            ],
            'events': [
                {
                    'recorded_at': NOW,
                    'version': 2,
                    'event': 'promoted',
                    'alias': 'champion',
                    'details': {'manuel': True},
                }
            ],
        }

    def _action(self, kind, **kwargs):
        self.calls.append((kind, kwargs))
        if self.refuse:
            raise api.ApiError('refusé par le test', 409)
        return {}

    def set_system(self, **kwargs):
        return self._action('system', **kwargs)

    def promote(self, name, **kwargs):
        return self._action('promote', name=name, **kwargs)

    def rollback(self, name):
        return self._action('rollback', name=name)


@pytest.fixture
def fake(monkeypatch):
    holder = {'client': FakeClient()}
    monkeypatch.setattr(api.ApiClient, 'from_env', classmethod(lambda cls, env=None: holder['client']))
    return holder


def _run(page_name: str = 'Projet', operator: str = '') -> AppTest:
    page = AppTest.from_file(PAGE, default_timeout=30)
    page.run()
    page.sidebar.radio(key='page').set_value(page_name).run()
    if operator:
        page.sidebar.selectbox(key='operator').set_value(operator).run()
    return page


def test_sidebar_offers_demo_operators_then_navigation(fake):
    page = _run()
    assert not page.exception
    operator = page.sidebar.selectbox(key='operator')
    assert operator.options == ['admin', 'analyste', 'default']
    assert operator.value == 'default'
    labels = page.sidebar.radio(key='page').options  # libellés affichés : icône puis nom
    assert [label.split(' ', 1)[1] for label in labels] == ['Projet', 'Service', 'Prévisions', 'Modèles']


def test_every_page_renders(fake):
    service = _run('Service')
    assert not service.exception
    assert [m.value for m in service.metric][:2] == ['ok', 'ok']
    assert any('en marche' in s.value for s in service.success)
    for name in ('Prévisions', 'Modèles'):
        page = _run(name)
        assert not page.exception
        assert page.dataframe


def test_unreachable_api_stops_with_hint(fake):
    fake['client'] = FakeClient(health_error=True)
    page = _run('Service')
    assert not page.exception
    assert 'docker compose up' in page.error[0].value
    assert not _run('Projet').error  # la présentation reste visible


def test_stop_requires_a_reason(fake):
    page = _run('Service')
    page.toggle[0].set_value(False)
    page.button(key='FormSubmitter:system-Appliquer').click().run()
    assert any('raison' in w.value for w in page.warning)
    assert fake['client'].calls == []


def test_stop_calls_api_with_trace(fake):
    page = _run('Service', 'admin')
    page.toggle[0].set_value(False)
    page.text_input(key='system_reason').input('maintenance')
    page.button(key='FormSubmitter:system-Appliquer').click().run()
    assert fake['client'].calls == [
        ('system', {'active': False, 'changed_by': 'admin', 'reason': 'maintenance'})
    ]
    assert any('Arrêt : fait' in s.value for s in page.success)


def test_promotion_defaults_to_challenger(fake):
    page = _run('Modèles', 'analyste')
    page.text_input(key='promote_reason').input('meilleur QLIKE')
    page.button(key='FormSubmitter:promote-Promouvoir en champion').click().run()
    assert fake['client'].calls == [
        (
            'promote',
            {'name': 'volatilite', 'version': 3, 'changed_by': 'analyste', 'reason': 'meilleur QLIKE'},
        )
    ]


def test_rollback_needs_confirmation_and_shows_refusal(fake):
    fake['client'] = FakeClient(refuse=True)
    page = _run('Modèles')
    submit = 'FormSubmitter:rollback-Revenir à la version précédente'
    page.button(key=submit).click().run()
    assert any('confirmation' in w.value for w in page.warning)
    assert fake['client'].calls == []
    page.checkbox[0].check()
    page.button(key=submit).click().run()
    assert fake['client'].calls == [('rollback', {'name': 'volatilite'})]
    assert any('refusé par le test' in e.value for e in page.error)
