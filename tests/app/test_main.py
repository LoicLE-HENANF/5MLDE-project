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


def _run(operator: str = '') -> AppTest:
    page = AppTest.from_file(PAGE, default_timeout=30)
    page.run()
    if operator:
        page.sidebar.text_input(key='operator').input(operator)
        page.run()
    return page


def test_page_renders_all_tabs(fake):
    page = _run()
    assert not page.exception
    assert [tab.label for tab in page.tabs] == ['Service', 'Prévisions', 'Modèles']
    assert [m.value for m in page.metric][:2] == ['ok', 'ok']
    assert len(page.dataframe) >= 3  # prévisions, santé, registre
    assert any('en marche' in s.value for s in page.tabs[0].success)


def test_unreachable_api_stops_with_hint(fake):
    fake['client'] = FakeClient(health_error=True)
    page = _run()
    assert not page.exception
    assert 'docker compose up' in page.error[0].value
    assert not page.tabs


def test_stop_requires_operator_and_reason(fake):
    page = _run()
    page.toggle[0].set_value(False)
    page.button(key='FormSubmitter:system-Appliquer').click().run()
    assert any("l'opérateur" in w.value for w in page.warning)
    assert fake['client'].calls == []


def test_stop_calls_api_with_trace(fake):
    page = _run('loic')
    page.toggle[0].set_value(False)
    page.text_input(key='system_reason').input('maintenance')
    page.button(key='FormSubmitter:system-Appliquer').click().run()
    assert fake['client'].calls == [
        ('system', {'active': False, 'changed_by': 'loic', 'reason': 'maintenance'})
    ]
    assert any('Arrêt : fait' in s.value for s in page.success)


def test_promotion_defaults_to_challenger(fake):
    page = _run('loic')
    page.text_input(key='promote_reason').input('meilleur QLIKE')
    page.button(key='FormSubmitter:promote-Promouvoir en champion').click().run()
    assert fake['client'].calls == [
        ('promote', {'name': 'volatilite', 'version': 3, 'changed_by': 'loic', 'reason': 'meilleur QLIKE'})
    ]


def test_rollback_needs_confirmation_and_shows_refusal(fake):
    fake['client'] = FakeClient(refuse=True)
    page = _run('loic')
    submit = 'FormSubmitter:rollback-Revenir à la version précédente'
    page.button(key=submit).click().run()
    assert any('confirmation' in w.value for w in page.warning)
    assert fake['client'].calls == []
    page.checkbox[0].check()
    page.button(key=submit).click().run()
    assert fake['client'].calls == [('rollback', {'name': 'volatilite'})]
    assert any('refusé par le test' in e.value for e in page.error)
