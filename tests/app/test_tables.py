"""Mise en forme des réponses de l'API."""

from __future__ import annotations

import pytest

from app import tables


@pytest.mark.parametrize(
    'build', [tables.predictions_frame, tables.health_frame, tables.events_frame, tables.system_frame]
)
def test_empty_rows_keep_columns(build):
    frame = build([])
    assert frame.empty and len(frame.columns) > 0


def test_predictions_frame_computes_gap_and_sorts():
    rows = [
        {
            'symbol': 'ETHUSDT',
            'open_time': '2026-11-18T10:00:00Z',
            'horizon_hours': 24,
            'prediction': 0.03,
            'baseline': 0.02,
            'position': 0.4,
            'version': 2,
        },
        {
            'symbol': 'BTCUSDT',
            'open_time': '2026-11-18T10:00:00Z',
            'horizon_hours': 24,
            'prediction': 0.02,
            'baseline': 0.02,
            'position': None,
            'version': 2,
        },
    ]
    frame = tables.predictions_frame(rows)
    assert frame['Symbole'].tolist() == ['BTCUSDT', 'ETHUSDT']
    assert frame['Écart'].tolist() == pytest.approx([0.0, 0.5])
    assert str(frame['Bougie'].dt.tz) == 'UTC'


def test_events_frame_translates_and_flattens_details():
    rows = [
        {
            'recorded_at': '2026-11-18T10:00:00Z',
            'version': 2,
            'event': 'promoted',
            'alias': None,
            'details': {'manuel': True, 'raisons': ['niveau 1 valide', 'niveau 2 échoue']},
        }
    ]
    frame = tables.events_frame(rows)
    assert frame.loc[0, 'Événement'] == 'promu'
    assert frame.loc[0, 'Alias'] == ''
    assert frame.loc[0, 'Détails'] == 'manuel | raisons : niveau 1 valide ; niveau 2 échoue'


def test_system_frame_labels_state():
    rows = [{'recorded_at': '2026-11-18T10:00:00Z', 'active': False, 'changed_by': 'loic', 'reason': 'test'}]
    assert tables.system_frame(rows).loc[0, 'État'] == 'arrêt'


def test_health_frame_sorted_by_day():
    base = {
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
    frame = tables.health_frame([{**base, 'day': '2026-11-18'}, {**base, 'day': '2026-11-17'}])
    assert [str(d) for d in frame['Jour']] == ['2026-11-17', '2026-11-18']
