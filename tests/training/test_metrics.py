"""Tests des métriques statistiques d'entraînement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import icir, information_coefficient


def _cross_section(
    timestamp: pd.Timestamp,
    predictions: list[float],
    targets: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'symbol': [f'SYMBOL{index}' for index in range(len(predictions))],
            'open_time': timestamp,
            'prediction': predictions,
            'target_return': targets,
        }
    )


def test_information_coefficient_returns_cross_sectional_spearman_per_timestamp():
    times = pd.date_range('2026-01-01', periods=2, freq='h', tz='UTC')
    frame = pd.concat(
        [
            _cross_section(times[0], [1, 2, 3, 4, 5], [10, 20, 30, 40, 50]),
            _cross_section(times[1], [1, 2, 3, 4, 5], [50, 40, 30, 20, 10]),
        ],
        ignore_index=True,
    )

    result = information_coefficient(frame)

    assert result.name == 'ic'
    assert result.index.equals(pd.DatetimeIndex(times, name='open_time'))
    np.testing.assert_allclose(result.to_numpy(), [1.0, -1.0])


def test_information_coefficient_drops_missing_pairs_and_marks_undefined_hours():
    times = pd.date_range('2026-01-01', periods=3, freq='h', tz='UTC')
    frame = pd.concat(
        [
            _cross_section(times[0], [1, 2, 3, 4], [10, 20, 30, np.nan]),
            _cross_section(times[1], [1, 2, 3], [10, 20, np.nan]),
            _cross_section(times[2], [1, 1, 1], [10, 20, 30]),
        ],
        ignore_index=True,
    )

    result = information_coefficient(frame)

    assert result.iloc[0] == pytest.approx(1.0)
    assert result.iloc[1:].isna().all()


def test_information_coefficient_rejects_duplicate_symbol_and_timestamp():
    frame = _cross_section(
        pd.Timestamp('2026-01-01', tz='UTC'),
        [1, 2, 3],
        [3, 2, 1],
    )
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match='doublons'):
        information_coefficient(duplicate)


def test_information_coefficient_requires_timezone_aware_timestamps():
    frame = _cross_section(
        pd.Timestamp('2026-01-01'),
        [1, 2, 3],
        [3, 2, 1],
    )

    with pytest.raises(ValueError, match='fuseau'):
        information_coefficient(frame)


def test_information_coefficient_rejects_infinite_values():
    frame = _cross_section(
        pd.Timestamp('2026-01-01', tz='UTC'),
        [1, 2, np.inf],
        [3, 2, 1],
    )

    with pytest.raises(ValueError, match='infinies'):
        information_coefficient(frame)


def test_icir_uses_sample_standard_deviation_and_ignores_nan():
    result = icir(pd.Series([0.1, 0.3, np.nan]))

    assert result == pytest.approx(np.sqrt(2))


@pytest.mark.parametrize(
    'values',
    [
        [0.5],
        [0.2, 0.2],
        [np.nan, np.nan],
        [0.2, np.nan],
    ],
)
def test_icir_is_nan_when_variability_or_observations_are_insufficient(values):
    assert np.isnan(icir(pd.Series(values, dtype='float64')))


def test_icir_rejects_non_finite_values():
    with pytest.raises(ValueError, match='infinis'):
        icir(pd.Series([0.1, np.inf]))
