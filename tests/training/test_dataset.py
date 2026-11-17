"""Tests de preparation des entrees du modele de rendement."""

from __future__ import annotations

import pandas as pd
import pytest

from features import FEATURE_COLUMNS
from marketdata.identity import identify
from marketdata.sources import normalize
from tests.conftest import raw_rows
from training import prepare_return_inputs


def _candles(rows: int = 180) -> pd.DataFrame:
    return normalize(raw_rows(rows), 'BTCUSDT')


def test_prepare_return_inputs_validates_identifies_and_uses_one_cutoff():
    candles = _candles()
    as_of = candles.loc[170, 'close_time'] + pd.Timedelta(microseconds=1)

    result = prepare_return_inputs(candles, as_of=as_of)

    assert result.as_of == as_of
    assert result.identity == identify(candles)
    assert result.quality_report.ok
    assert result.features.columns.tolist() == ['symbol', 'open_time', *FEATURE_COLUMNS]
    assert result.targets.columns.tolist() == ['symbol', 'open_time', 'target_return']
    assert len(result.features) == 171
    assert result.features['open_time'].max() == candles.loc[170, 'open_time']
    assert (
        result.targets.loc[result.targets['target_return'].notna(), 'open_time'].max()
        == candles.loc[166, 'open_time']
    )


def test_prepare_return_inputs_rejects_blocking_quality_issues():
    candles = _candles()
    candles.loc[0, 'open'] = -1

    with pytest.raises(ValueError, match='jeu de bougies invalide'):
        prepare_return_inputs(candles)


def test_prepare_return_inputs_requires_timezone_aware_as_of():
    with pytest.raises(ValueError, match='fuseau horaire'):
        prepare_return_inputs(_candles(), as_of=pd.Timestamp('2025-01-02'))
