"""Règle de promotion et cycle d'alias champion / challenger / précédent sur un vrai MLflow local."""

from __future__ import annotations

import mlflow
import mlflow.pyfunc
import pytest

from registry import CHALLENGER, CHAMPION, PREVIOUS, RegistryError, decide, publish, rollback, status
from tests.connectors.test_mlflow import _Constant

NAME = 'volatilite'

# -- décision ---------------------------------------------------------------------------------


def test_level1_failure_is_not_registered():
    decision = decide(level1=False, level2=True, score=0.1, champion_score=None)
    assert decision.alias is None
    assert not decision.registered


def test_level2_failure_is_challenger_only():
    decision = decide(level1=True, level2=False, score=0.1, champion_score=None)
    assert decision.alias == CHALLENGER


def test_first_model_passing_both_levels_becomes_champion():
    assert decide(level1=True, level2=True, score=0.3, champion_score=None).alias == CHAMPION


def test_must_strictly_beat_current_champion():
    assert decide(level1=True, level2=True, score=0.2, champion_score=0.3).alias == CHAMPION
    assert decide(level1=True, level2=True, score=0.3, champion_score=0.3).alias == CHALLENGER
    assert decide(level1=True, level2=True, score=0.4, champion_score=0.3).alias == CHALLENGER


# -- publication et retour arrière ------------------------------------------------------------


def _log(connector):
    with connector.start_run('train'):
        return connector.log_model(mlflow.pyfunc.log_model, python_model=_Constant())


class Recorder:
    def __init__(self):
        self.events: list[dict] = []

    def __call__(self, **kwargs):
        self.events.append(kwargs)


def test_rejected_decision_registers_nothing(local_connector, released):
    info = _log(local_connector)
    decision = decide(level1=False, level2=False, score=1.0, champion_score=None)
    assert publish(local_connector, info, NAME, decision) is None
    assert local_connector.client.search_registered_models() == []


def test_promotion_cycle_and_rollback(local_connector, released):
    record = Recorder()
    champion = decide(level1=True, level2=True, score=0.3, champion_score=None)
    first = publish(local_connector, _log(local_connector), NAME, champion, record=record)
    assert status(local_connector, NAME) == {CHAMPION: str(first.version), CHALLENGER: None, PREVIOUS: None}

    better = decide(level1=True, level2=True, score=0.2, champion_score=0.3)
    second = publish(local_connector, _log(local_connector), NAME, better, record=record)
    assert status(local_connector, NAME) == {
        CHAMPION: str(second.version),
        CHALLENGER: None,
        PREVIOUS: str(first.version),
    }

    worse = decide(level1=True, level2=True, score=0.5, champion_score=0.2)
    third = publish(local_connector, _log(local_connector), NAME, worse, record=record)
    assert status(local_connector, NAME)[CHALLENGER] == str(third.version)
    assert status(local_connector, NAME)[CHAMPION] == str(second.version)

    restored = rollback(local_connector, NAME, record=record)
    assert str(restored.version) == str(first.version)
    assert status(local_connector, NAME) == {
        CHAMPION: str(first.version),
        CHALLENGER: str(third.version),
        PREVIOUS: None,
    }

    events = [(e['version'], e['event'], e.get('alias')) for e in record.events]
    assert events[:2] == [(first.version, 'registered', None), (first.version, 'promoted', CHAMPION)]
    assert (first.version, 'retired', CHAMPION) in events
    assert events[-2:] == [(second.version, 'retired', CHAMPION), (first.version, 'promoted', CHAMPION)]
    assert all(e['model_uri'] == f'models:/{NAME}/{e["version"]}' for e in record.events)


def test_rollback_without_previous_fails(local_connector, released):
    publish(
        local_connector,
        _log(local_connector),
        NAME,
        decide(level1=True, level2=True, score=0.3, champion_score=None),
    )
    with pytest.raises(RegistryError, match='precedent'):
        rollback(local_connector, NAME)
