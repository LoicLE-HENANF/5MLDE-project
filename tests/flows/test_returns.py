"""Tests de l'orchestration production du modèle rendement."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

from flows.feedback import RETURN_FLOW_NAME, score_pending
from flows.predict import forecast
from flows.serve import SCHEDULES
from serving import Forecast
from training.returns import MODEL_NAME


def test_return_model_flows_are_deployed():
    deployed = {flow.name for flow, _ in SCHEDULES.values()}

    assert {'train-returns', 'predict-returns', RETURN_FLOW_NAME} <= deployed


def test_return_forecast_uses_signed_return_inference(monkeypatch):
    import flows.predict as predict_flow

    stamp = pd.Timestamp('2026-01-01', tz='UTC')
    version = SimpleNamespace(version=3, run_id='run-1')
    frame = pd.DataFrame(
        {
            'symbol': ['BTCUSDT'],
            'open_time': [stamp],
            'prediction': [-0.01],
            'baseline': [0.02],
            'position': [-1.0],
        }
    )
    expected = Forecast(frame=frame, as_of=stamp)
    calls = {}

    def infer(predict, candles, *, as_of, with_positions):
        calls.update(as_of=as_of, with_positions=with_positions)
        return expected

    monkeypatch.setattr(predict_flow, 'predict_latest_returns', infer)
    connector = SimpleNamespace(
        get_version=lambda model, alias: version,
        load_model=lambda model, alias, loader: SimpleNamespace(predict=lambda rows: [-0.01]),
    )

    actual_version, result = forecast.fn(
        connector,
        'champion',
        pd.DataFrame(),
        datetime(2026, 1, 1, tzinfo=UTC),
        MODEL_NAME,
        4,
    )

    assert actual_version is version
    assert result is expected
    assert calls['with_positions'] is True
    assert calls['as_of'] == datetime(2026, 1, 1, tzinfo=UTC)


def test_return_feedback_reads_and_records_only_return_predictions(monkeypatch):
    import flows.feedback as feedback_flow

    timestamp = pd.Timestamp('2026-01-01', tz='UTC')
    waiting = [
        {
            'prediction_id': 10,
            'symbol': 'BTCUSDT',
            'open_time': timestamp,
            'horizon_hours': 4,
            'prediction': -0.01,
            'baseline': 0.02,
        }
    ]
    scored = pd.DataFrame([{'prediction_id': 10, 'realized': -0.02, 'loss': 0.0001, 'baseline_loss': 0.0016}])
    db = MagicMock()
    db.select.return_value = waiting
    monkeypatch.setattr(feedback_flow, 'realized_return_feedback', lambda *args, **kwargs: scored)

    count = score_pending.fn(
        db,
        pd.DataFrame(),
        timestamp.to_pydatetime() + timedelta(hours=6),
        timestamp.to_pydatetime(),
        None,
        MODEL_NAME,
    )

    assert count == 1
    assert db.select.call_args.args[1] == [MODEL_NAME, timestamp.to_pydatetime()]
    db.record_feedback.assert_called_once_with(
        prediction_id=10,
        realized=-0.02,
        loss=0.0001,
        baseline_loss=0.0016,
        prefect_flow_run_id=None,
    )
