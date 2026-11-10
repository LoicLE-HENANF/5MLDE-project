"""Commande de vérification : agrégation des résultats, code de sortie."""

from __future__ import annotations

import pytest

from connectors import health
from connectors.errors import ConfigError, ConnectorError


@pytest.fixture
def checks(monkeypatch):
    outcomes: dict[str, Exception | None] = {name: None for name in health.CHECKS}

    def fake(name):
        def check(env):
            if outcomes[name] is not None:
                raise outcomes[name]

        return check

    monkeypatch.setattr(health, 'CHECKS', {name: fake(name) for name in health.CHECKS})
    monkeypatch.setattr(health, 'load_env', lambda: {})
    return outcomes


def test_all_ok(checks, capsys):
    assert health.main([]) == 0
    out = capsys.readouterr().out
    assert all(f'[OK] {name}' in out for name in checks)


def test_one_failure_sets_exit_code(checks, capsys):
    checks['postgres'] = ConnectorError('PostgreSQL', 'x', 'connexion impossible')
    checks['mlflow'] = ConfigError('MLFLOW_TRACKING_URI : variable absente')
    assert health.main([]) == 1
    out = capsys.readouterr().out
    assert '[KO] postgres' in out and 'connexion impossible' in out
    assert '[KO] mlflow' in out and '[OK] grafana' in out


def test_subset(checks, capsys):
    assert health.main(['grafana']) == 0
    assert capsys.readouterr().out.strip() == '[OK] grafana'


def test_unknown_service(checks):
    with pytest.raises(SystemExit):
        health.main(['redis'])


def test_unexpected_errors_are_not_hidden(checks):
    checks['grafana'] = KeyError('bug')
    with pytest.raises(KeyError):
        health.main([])
