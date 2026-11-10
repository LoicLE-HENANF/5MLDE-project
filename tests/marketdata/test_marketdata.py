"""Définition des données : sources, schéma, règles qualité, identité (sans réseau)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pandera.errors
import pytest

from marketdata import identity, quality, schema, sources
from marketdata.universe import SYMBOLS, check_symbol
from tests.conftest import HOUR_MS, START_MS, raw_rows


def candles(n=3, symbol='BTCUSDT', **kwargs):
    return sources.normalize(raw_rows(n, **kwargs), symbol)


# -- univers et URLs -------------------------------------------------------------------


def test_check_symbol():
    assert check_symbol('ETHUSDT') == 'ETHUSDT'
    with pytest.raises(ValueError, match='hors univers'):
        check_symbol('DOGEUSDT')


def test_archive_and_checksum_urls():
    url = sources.archive_url('BTCUSDT', date(2024, 3, 1))
    assert url == ('https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-03.zip')
    assert sources.checksum_url('BTCUSDT', date(2024, 3, 1)) == url + '.CHECKSUM'


def test_rest_params():
    params = sources.rest_params('SOLUSDT', 1, 2)
    assert params == {'symbol': 'SOLUSDT', 'interval': '1h', 'startTime': 1, 'limit': 1000, 'endTime': 2}
    assert 'endTime' not in sources.rest_params('SOLUSDT', 1)


def test_parse_checksum():
    digest = 'a' * 64
    assert sources.parse_checksum(f'{digest.upper()}  BTCUSDT-1h-2024-03.zip\n') == digest
    for bad in ('', 'zz  x.zip', 'a' * 63):
        with pytest.raises(sources.SourceFormatError):
            sources.parse_checksum(bad)


def test_read_archive_csv_skips_header():
    body = '\n'.join(','.join(row) for row in raw_rows(2))
    header = ','.join(sources.RAW_FIELDS)
    assert sources.read_archive_csv(body.encode()) == raw_rows(2)
    assert sources.read_archive_csv(f'{header}\n{body}\n') == raw_rows(2)


# -- normalisation ---------------------------------------------------------------------


def test_normalize_matches_schema():
    frame = candles()
    schema.validate(frame)
    assert tuple(frame.columns) == schema.COLUMNS
    assert frame['open_time'].iloc[0] == pd.Timestamp('2025-01-01T00:00:00Z')
    assert frame['trades'].tolist() == [42, 42, 42]


def test_milliseconds_and_microseconds_give_same_candles():
    pd.testing.assert_frame_equal(candles(micros=False), candles(micros=True))


def test_mixed_units_within_one_batch():
    rows = raw_rows(1, micros=False) + raw_rows(1, start_ms=START_MS + HOUR_MS, micros=True)
    frame = sources.normalize(rows, 'BTCUSDT')
    assert frame['open_time'].diff().iloc[1] == pd.Timedelta(hours=1)


def test_close_time_is_canonical():
    frame = candles(1)
    assert frame['close_time'].iloc[0] == pd.Timestamp('2025-01-01T00:59:59.999999Z')


def test_normalize_sorts_rows():
    frame = sources.normalize(list(reversed(raw_rows(3))), 'BTCUSDT')
    assert frame['open_time'].is_monotonic_increasing


@pytest.mark.parametrize(
    ('mutate', 'match'),
    [
        (lambda row: row.pop(), '12 champs'),
        (lambda row: row.__setitem__(1, 'abc'), 'non numérique'),
        (lambda row: row.__setitem__(6, str(int(row[0]) + 60_000 - 1)), 'close_time'),
    ],
)
def test_normalize_rejects_bad_rows(mutate, match):
    rows = raw_rows(2)
    mutate(rows[1])
    with pytest.raises(sources.SourceFormatError, match=match):
        sources.normalize(rows, 'BTCUSDT')


def test_normalize_rejects_unknown_symbol():
    with pytest.raises(ValueError, match='hors univers'):
        sources.normalize(raw_rows(1), 'DOGEUSDT')


# -- schéma ----------------------------------------------------------------------------


def test_schema_rejects_bad_values():
    frame = candles()
    frame.loc[0, 'open'] = -1.0
    frame.loc[1, 'trades'] = -3
    with pytest.raises(pandera.errors.SchemaErrors) as info:
        schema.validate(frame)
    assert set(info.value.failure_cases['column']) == {'open', 'trades'}


def test_schema_is_strict():
    with pytest.raises(pandera.errors.SchemaErrors):
        schema.validate(candles().assign(extra=1))


# -- qualité ---------------------------------------------------------------------------


def rules(report):
    return {issue.rule: issue for issue in report.issues}


def test_clean_batch_passes():
    report = quality.check(pd.concat([candles(24, symbol=s) for s in SYMBOLS], ignore_index=True))
    assert report.ok
    assert report.issues == ()
    assert report.summary() == '120 bougies, aucune anomalie'


def test_schema_failure_stops_other_rules():
    frame = candles()
    frame.loc[0, 'low'] = 0.0
    report = quality.check(pd.concat([frame, frame]))
    assert not report.ok
    assert list(rules(report)) == ['schéma']


def test_duplicates_block():
    frame = candles()
    report = quality.check(pd.concat([frame, frame.iloc[[1]]], ignore_index=True))
    issue = rules(report)['doublons (symbol, open_time)']
    assert issue.blocking and issue.count == 2
    assert issue.examples[0] == 'BTCUSDT 2025-01-01 01:00'


def test_misaligned_blocks():
    frame = candles()
    frame.loc[1, 'open_time'] += pd.Timedelta(minutes=30)
    assert "open_time non aligné sur l'heure" in rules(quality.check(frame))


@pytest.mark.parametrize(('column', 'value'), [('high', 99.0), ('low', 102.0)])
def test_ohlc_incoherence_blocks(column, value):
    frame = candles()
    frame.loc[0, column] = value
    report = quality.check(frame)
    assert not report.ok
    assert rules(report)['incohérence OHLC'].count == 1


def test_taker_volume_above_total_blocks():
    frame = candles()
    frame.loc[2, 'taker_buy_base_volume'] = 11.0
    assert not quality.check(frame).ok


def test_small_gap_is_reported_not_blocking():
    frame = candles(200).drop(index=[50])
    report = quality.check(frame)
    issue = rules(report)['bougies manquantes']
    assert report.ok and not issue.blocking
    assert issue.examples == ('BTCUSDT 2025-01-03 02:00',)


def test_large_gap_blocks():
    frame = candles(50).drop(index=range(10, 20))
    issue = rules(quality.check(frame))['bougies manquantes']
    assert issue.blocking and issue.count == 10


def test_gaps_are_per_symbol():
    eth = candles(3, symbol='ETHUSDT', start_ms=START_MS + 100 * HOUR_MS)
    assert quality.check(pd.concat([candles(3), eth], ignore_index=True)).ok


def test_returns():
    frame = candles(4)
    frame.loc[1, 'close'] = 130.0  # ln(130/101) = 0.25 puis ln(103/130) = -0.23
    frame.loc[1, 'high'] = 131.0
    report = quality.check(frame)
    assert report.ok
    assert rules(report)['variation horaire > 20%'].count == 2

    frame.loc[1, ['close', 'high']] = 200.0
    report = quality.check(frame)
    assert not report.ok
    assert rules(report)['variation horaire > 50%'].count == 2


def test_return_ignores_non_consecutive_candles():
    frame = candles(40)
    frame.loc[21:, ['open', 'high', 'low', 'close']] *= 3
    frame = frame.drop(index=20)
    assert 'variation horaire > 50%' not in rules(quality.check(frame))


def test_summary_lists_issues():
    frame = candles()
    frame.loc[0, 'high'] = 50.0
    assert '[BLOQUANT] incohérence OHLC : 1 (ex. BTCUSDT 2025-01-01 00:00)' in quality.check(frame).summary()


# -- identité --------------------------------------------------------------------------


def test_identity():
    frame = pd.concat([candles(5, symbol='ETHUSDT'), candles(5)], ignore_index=True)
    ident = identity.identify(frame)
    assert ident.symbols == ('BTCUSDT', 'ETHUSDT')
    assert ident.start == datetime(2025, 1, 1, tzinfo=UTC)
    assert ident.end == datetime(2025, 1, 1, 4, tzinfo=UTC)
    assert ident.rows == 10
    assert len(ident.sha256) == 64
    assert ident.tags()['data.sha256'] == ident.sha256
    assert ident.tags()['data.symbols'] == 'BTCUSDT,ETHUSDT'


def test_fingerprint_ignores_row_order_and_source_unit():
    frame = candles(5)
    shuffled = candles(5, micros=True).sample(frac=1, random_state=0)
    assert identity.fingerprint(frame) == identity.fingerprint(shuffled)


def test_fingerprint_detects_any_change():
    frame = candles(5)
    changed = frame.copy()
    changed.loc[3, 'volume'] += 1e-8
    assert identity.fingerprint(frame) != identity.fingerprint(changed)


def test_identity_rejects_empty_or_invalid():
    with pytest.raises(ValueError, match='vide'):
        identity.identify(candles().iloc[0:0])
    with pytest.raises(pandera.errors.SchemaErrors):
        identity.identify(candles().assign(open=-1.0))
