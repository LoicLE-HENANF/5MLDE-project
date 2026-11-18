"""Positions simulées et backtest horaire."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies import (
    backtest,
    equal_weight,
    long_cash,
    long_short,
    next_returns,
    scale_by_volatility,
    staggered,
    to_wide,
)

T0 = pd.Timestamp('2025-01-01', tz='UTC')
HOUR = pd.Timedelta(hours=1)


def _predictions(rows: list[tuple[str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'symbol': [s for s, _, _ in rows],
            'open_time': [T0 + h * HOUR for _, h, _ in rows],
            'prediction': [p for _, _, p in rows],
        }
    )


def _candles(closes: dict[str, list[float]]) -> pd.DataFrame:
    frames = [
        pd.DataFrame({'symbol': symbol, 'open_time': T0 + np.arange(len(values)) * HOUR, 'close': values})
        for symbol, values in closes.items()
    ]
    return pd.concat(frames, ignore_index=True)


# -- positions ---------------------------------------------------------------------------------
def test_staggered_averages_the_last_hours_and_cuts_turnover():
    signs = [1, -1, 1, -1, 1, -1, 1, -1]
    rows = [(s, h, sign * (1 if s == 'A' else -1)) for h, sign in enumerate(signs) for s in ('A', 'B')]
    hourly = long_short(_predictions(rows))
    held = staggered(hourly, 4)

    assert held.loc[T0, 'A'] == pytest.approx(0.5)  # une seule heure connue
    assert held.loc[T0 + 3 * HOUR, 'A'] == pytest.approx(0.0)  # deux heures longues, deux courtes
    assert (held.sum(axis=1).abs() < 1e-12).all()  # reste neutre au marché
    assert (held.abs().sum(axis=1) <= 1 + 1e-12).all()
    assert held.diff().abs().sum().sum() < hourly.diff().abs().sum().sum() / 2
    assert staggered(hourly, 1).equals(hourly.sort_index())


def test_staggered_rejects_bad_inputs():
    hourly = long_short(_predictions([('A', 0, 0.1), ('B', 0, 0.2)]))
    for hours in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match='entier positif'):
            staggered(hourly, hours)
    with pytest.raises(ValueError, match='indexés par heure'):
        staggered(hourly.reset_index(drop=True), 4)


def test_to_wide_rejects_bad_inputs():
    good = _predictions([('A', 0, 0.1)])
    with pytest.raises(ValueError, match='absentes'):
        to_wide(good.drop(columns='prediction'))
    with pytest.raises(ValueError, match='fuseau'):
        to_wide(good.assign(open_time=good['open_time'].dt.tz_localize(None)))
    with pytest.raises(ValueError, match='doublons'):
        to_wide(pd.concat([good, good]))
    with pytest.raises(ValueError, match='infinies'):
        to_wide(good.assign(prediction=np.inf))


def test_long_short_is_market_neutral_with_unit_gross_exposure():
    weights = long_short(_predictions([('A', 0, 0.03), ('B', 0, -0.01), ('C', 0, 0.0)]))
    row = weights.iloc[0]
    assert row['A'] > 0 > row['B']
    assert row['C'] == pytest.approx(0.0)
    assert row.sum() == pytest.approx(0.0)
    assert row.abs().sum() == pytest.approx(1.0)


def test_long_short_stays_in_cash_without_signal():
    weights = long_short(_predictions([('A', 0, 0.01), ('B', 0, 0.01), ('A', 1, 0.02), ('C', 2, 0.5)]))
    assert (weights.to_numpy() == 0).all()  # égalité, puis un seul symbole prédit par heure


def test_long_cash_buys_only_above_the_fee_threshold():
    weights = long_cash(_predictions([('A', 0, 0.005), ('B', 0, 0.0005), ('C', 0, -0.02)]))
    assert weights.iloc[0].tolist() == pytest.approx([1 / 3, 0.0, 0.0])
    assert long_cash(_predictions([('A', 0, 0.0005)]), threshold=0.0).iloc[0, 0] == 1.0


def test_equal_weight():
    weights = equal_weight(pd.Index([T0]), ['B', 'A'])
    assert weights.columns.tolist() == ['A', 'B']
    assert weights.iloc[0].tolist() == [0.5, 0.5]
    with pytest.raises(ValueError):
        equal_weight(pd.Index([T0]), [])


def test_scale_by_volatility_targets_risk_without_leverage():
    positions = pd.DataFrame({'A': [0.5, 0.5], 'B': [-0.5, 0.5]}, index=[T0, T0 + HOUR])
    volatility = pd.DataFrame({'A': [0.02, 0.005], 'B': [0.04, np.nan]}, index=[T0, T0 + HOUR])
    scaled = scale_by_volatility(positions, volatility, target=0.01)
    assert scaled.loc[T0].tolist() == pytest.approx([0.25, -0.125])
    assert scaled.loc[T0 + HOUR].tolist() == pytest.approx([0.5, 0.0])  # plafond 1, puis cash
    with pytest.raises(ValueError):
        scale_by_volatility(positions, volatility, target=0)
    with pytest.raises(ValueError, match='positives'):
        scale_by_volatility(positions, volatility.fillna(0.0), target=0.01)


# -- backtest ----------------------------------------------------------------------------------
def test_next_returns_uses_the_following_close_and_marks_gaps():
    candles = _candles({'A': [100.0, 110.0, 99.0]}).drop(index=1)
    returns = next_returns(candles)
    assert len(returns) == 3  # l'heure manquante est réinsérée
    assert returns['A'].isna().all()  # aucune paire de bougies consécutives


def test_backtest_applies_position_to_the_next_hour_and_charges_fees():
    candles = _candles({'A': [100.0, 110.0, 121.0, 121.0], 'B': [50.0, 50.0, 40.0, 40.0]})
    positions = pd.DataFrame(
        {'A': [1.0, 0.0, 0.0], 'B': [0.0, -1.0, 0.0]},
        index=pd.DatetimeIndex([T0, T0 + HOUR, T0 + 2 * HOUR], name='open_time'),
    )
    result = backtest(positions, candles, fee=0.001)
    frame = result.frame
    assert frame['gross'].tolist() == pytest.approx([0.10, 0.20, 0.0])
    assert frame['turnover'].tolist() == pytest.approx([1.0, 2.0, 1.0])
    assert frame['net'].tolist() == pytest.approx([0.099, 0.198, -0.001])
    assert result.metrics['total_return'] == pytest.approx(1.099 * 1.198 * 0.999 - 1)
    assert result.metrics['fees'] == pytest.approx(0.004)
    assert result.metrics['max_drawdown'] == pytest.approx(-0.001)


def test_backtest_drops_trailing_hours_without_outcome():
    candles = _candles({'A': [100.0, 101.0]})
    positions = pd.DataFrame({'A': [1.0, 1.0]}, index=[T0, T0 + HOUR])
    result = backtest(positions, candles, fee=0.0)
    assert len(result.frame) == 1
    assert result.metrics['total_return'] == pytest.approx(0.01)


def test_backtest_rejects_invalid_inputs():
    candles = _candles({'A': [100.0, 101.0]})
    positions = pd.DataFrame({'A': [1.0]}, index=[T0])
    with pytest.raises(ValueError, match='fee'):
        backtest(positions, candles, fee=-0.1)
    with pytest.raises(ValueError, match='sans bougies'):
        backtest(positions.rename(columns={'A': 'Z'}), candles)
    with pytest.raises(ValueError, match='finies'):
        backtest(positions.assign(A=np.nan), candles)
    with pytest.raises(ValueError, match='aucun rendement'):
        backtest(pd.DataFrame({'A': [1.0]}, index=[T0 + 5 * HOUR]), candles)


def test_volatility_targeting_beats_raw_exposure_on_risk():
    rng = np.random.default_rng(0)
    hours = 2000
    sigma = np.where(np.arange(hours) % 400 < 200, 0.002, 0.02)  # régimes calme / agité
    close = 100 * np.exp(np.cumsum(rng.normal(0, sigma)))
    candles = _candles({'A': close.tolist()})
    index = pd.DatetimeIndex(T0 + np.arange(hours) * HOUR)
    raw = pd.DataFrame({'A': 1.0}, index=index)
    forecast = pd.DataFrame({'A': np.roll(sigma, -1)}, index=index)  # oracle de la volatilité suivante
    targeted = scale_by_volatility(raw, forecast, target=0.002)
    raw_drawdown = backtest(raw, candles, fee=0.0).metrics['max_drawdown']
    targeted_drawdown = backtest(targeted, candles, fee=0.0).metrics['max_drawdown']
    assert targeted_drawdown > raw_drawdown
