"""Incidents contrôlés : corruption du lot et enchaînement des scénarios (sans la stack)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from flows import ingest
from incident import scenarios
from marketdata import quality, sources

OPEN_MS = 1_735_689_600_000  # 2025-01-01 00:00 UTC
HOUR_MS = 3_600_000


def _rows(n: int = 3) -> list[list[str | int]]:
    rows = []
    for i in range(n):
        start = OPEN_MS + i * HOUR_MS
        rows.append(
            [
                start,
                '100.0',
                '101.0',
                '99.0',
                '100.5',
                '10.0',
                start + HOUR_MS - 1,
                '1000.0',
                5,
                '4.0',
                '400.0',
                '0',
            ]
        )
    return rows


END = datetime(2025, 1, 1, 3, tzinfo=UTC)  # les 3 bougies sont clôturées


def _closed(rows):
    frame, _ = sources.normalize_dropping_interrupted(rows, 'BTCUSDT')
    return frame


def test_corrupt_breaks_only_the_last_closed_candle_and_is_blocking():
    rows = _rows()
    bad = scenarios.corrupt(rows, END)
    assert bad[:-1] == rows[:-1] and rows[-1][2] == '101.0'  # copie, original intact
    assert quality.check(_closed(rows)).ok
    report = quality.check(_closed(bad))
    assert not report.ok
    assert any(i.rule == 'incohérence OHLC' and i.blocking and i.count == 1 for i in report.issues)


def test_corrupt_skips_the_candle_in_progress():
    rows = _rows()
    bad = scenarios.corrupt(rows, END - timedelta(minutes=30))  # la 3e bougie est en cours
    assert bad[2] == rows[2] and bad[1][2] != rows[1][2]


def test_corrupt_refuses_a_batch_without_closed_candle():
    with pytest.raises(RuntimeError, match='aucune bougie'):
        scenarios.corrupt([], END)
    with pytest.raises(RuntimeError, match='aucune bougie'):
        scenarios.corrupt(_rows(1), datetime(2025, 1, 1, 0, 30, tzinfo=UTC))


def test_lot_invalide_reports_the_rejection(monkeypatch):
    def flow(symbols, history):
        assert ingest.BinanceClient is scenarios.CorruptedClient
        assert (symbols, history) == ([scenarios.SYMBOL], False)
        raise ingest.IngestionFailed('lots rejetés')

    monkeypatch.setattr(ingest, 'ingest_candles', flow)
    assert scenarios.lot_invalide(log=lambda _: None)
    assert ingest.BinanceClient is not scenarios.CorruptedClient


def test_lot_invalide_fails_when_the_batch_is_accepted(monkeypatch):
    monkeypatch.setattr(ingest, 'ingest_candles', lambda **_: {'rejected': 0})
    assert not scenarios.lot_invalide(log=lambda _: None)


class _Api:
    def __init__(self, states):
        self.states = iter(states)
        self.closed = False

    def health(self):
        return {'status': next(self.states), 'services': {}}

    def close(self):
        self.closed = True


def _dependance(monkeypatch, states):
    api = _Api(states)
    calls = []
    monkeypatch.setattr(scenarios.ApiClient, 'from_env', classmethod(lambda cls: api))
    monkeypatch.setattr(scenarios, '_compose', lambda *args: calls.append(args))
    monkeypatch.setattr(scenarios.time, 'sleep', lambda _: None)
    ok = scenarios.dependance('mlflow', pause=lambda: None, log=lambda _: None)
    return ok, calls, api


def test_dependance_stops_then_restarts_and_checks_recovery(monkeypatch):
    ok, calls, api = _dependance(monkeypatch, ['ok', 'ok', 'degraded', 'degraded', 'ok'])
    assert ok
    assert calls == [('stop', 'mlflow'), ('start', 'mlflow')]
    assert api.closed


def test_dependance_restarts_even_if_the_pause_is_interrupted(monkeypatch):
    api = _Api(['ok', 'degraded'])
    calls = []
    monkeypatch.setattr(scenarios.ApiClient, 'from_env', classmethod(lambda cls: api))
    monkeypatch.setattr(scenarios, '_compose', lambda *args: calls.append(args))

    def interrupted():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        scenarios.dependance('mlflow', pause=interrupted, log=lambda _: None)
    assert calls[-1] == ('start', 'mlflow') and api.closed
