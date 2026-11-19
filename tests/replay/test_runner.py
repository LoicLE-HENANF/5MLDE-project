"""Boucle du rejeu sur un vrai MLflow local : registre séparé, règle du rejeu, retour arrière, reprise."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from types import SimpleNamespace

import mlflow
import mlflow.pyfunc
import pandas as pd

from features import FEATURE_COLUMNS
from registry.promotion import CHALLENGER, CHAMPION, PREVIOUS, status
from replay import models
from replay.runner import LEVEL2_FAILED, Replay, replay_decision
from tests.connectors.test_mlflow import _Constant, local_connector, released  # noqa: F401
from tests.serving.test_serving import _candles

AS_OF = pd.Timestamp('2025-01-12', tz='UTC')


class FakeStore:
    """Mémoire à la place de PostgreSQL : seules les méthodes utilisées par ces tests."""

    def __init__(self, done=(), degraded_days=0):
        self.events = []
        self.done = set(done)
        self.degraded_days = degraded_days

    def record_event(self, **event):
        self.events.append(event)

    def done_months(self, model_name):
        return self.done

    def health_history(self, model_name, version, metric, until, limit):
        return [{'day': until - timedelta(days=i), 'degraded': i < self.degraded_days} for i in range(limit)]


def _spec(results):
    """Spec minimal : chaque entraînement renvoie le résultat suivant de `results`."""
    queue = iter(results)
    current = {}

    def fit(candles, as_of):
        current.update(next(queue))
        return SimpleNamespace(**current)

    def log(result, connector):
        with connector.start_run('train') as run:
            info = connector.log_model(mlflow.pyfunc.log_model, python_model=_Constant())
        return run.info.run_id, info

    base = models.spec('volatilite')
    return replace(
        base,
        fit=fit,
        log=log,
        level1=lambda result: result.level1,
        level2=lambda result: result.level2,
        score=lambda result: result.score,
        holdout=lambda result: pd.DataFrame({c: [0.0] for c in FEATURE_COLUMNS}),
        champion_score=lambda predicted, holdout: current['champion'],
    )


def _replay(connector, spec, store):
    return Replay(spec, connector, store, _candles(), log=lambda message: None)


def test_replay_decision_traces_level2_without_blocking():
    decision = replay_decision(level1=True, level2=False, score=0.1, champion_score=0.2)

    assert decision.alias == CHAMPION
    assert LEVEL2_FAILED in decision.reasons
    assert replay_decision(level1=False, level2=True, score=0.1, champion_score=None).alias is None
    assert replay_decision(level1=True, level2=False, score=0.3, champion_score=0.2).alias == CHALLENGER


def test_train_uses_the_replay_registry_and_leaves_live_untouched(local_connector, released):  # noqa: F811
    spec = _spec(
        [
            {'level1': True, 'level2': False, 'score': 0.5, 'champion': None},
            {'level1': True, 'level2': False, 'score': 0.6, 'champion': 0.5},
            {'level1': True, 'level2': True, 'score': 0.4, 'champion': 0.5},
        ]
    )
    store = FakeStore()
    replay = _replay(local_connector, spec, store)

    for month in range(3):
        replay.train(AS_OF + pd.DateOffset(months=month))

    assert status(local_connector, 'volatilite-replay') == {CHAMPION: '3', CHALLENGER: '2', PREVIOUS: '1'}
    names = {model.name for model in local_connector.client.search_registered_models()}
    assert names == {'volatilite-replay'}
    assert {event['model_name'] for event in store.events} == {'volatilite-replay'}


def test_rollback_after_consecutive_degraded_days(local_connector, released):  # noqa: F811
    spec = _spec(
        [
            {'level1': True, 'level2': True, 'score': 0.5, 'champion': None},
            {'level1': True, 'level2': True, 'score': 0.4, 'champion': 0.5},
        ]
    )
    store = FakeStore(degraded_days=3)
    replay = _replay(local_connector, spec, store)
    replay.train(AS_OF)
    replay.train(AS_OF + pd.DateOffset(months=1))

    assert replay._rollback(date(2025, 2, 20)) == 1
    assert status(local_connector, 'volatilite-replay')[CHAMPION] == '1'


def test_no_rollback_while_health_is_fine(local_connector, released):  # noqa: F811
    spec = _spec([{'level1': True, 'level2': True, 'score': 0.5, 'champion': None}])
    replay = _replay(local_connector, spec, FakeStore(degraded_days=2))
    replay.train(AS_OF)

    assert replay._rollback(date(2025, 1, 20)) is None


def test_run_skips_months_already_replayed(local_connector):  # noqa: F811
    spec = _spec([])
    replay = _replay(local_connector, spec, FakeStore(done={'2025-01', '2025-02'}))

    assert replay.run('2025-01', 2) == []
