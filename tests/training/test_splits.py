"""Tests du découpage walk-forward expansif."""

from __future__ import annotations

import pandas as pd
import pytest

from training import holdout_split, walk_forward_splits


@pytest.fixture
def panel() -> pd.DataFrame:
    times = pd.date_range('2022-01-01', '2023-06-30 23:00', freq='h', tz='UTC')
    return pd.concat(
        [
            pd.DataFrame({'symbol': symbol, 'open_time': times, 'target': 0.01})
            for symbol in ('BTCUSDT', 'ETHUSDT')
        ],
        ignore_index=True,
    )


def _times(frame: pd.DataFrame, indices) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(frame.iloc[indices]['open_time'])


def test_splits_use_expanding_train_and_monthly_validation_windows(panel):
    folds = walk_forward_splits(panel, target_column='target')

    assert len(folds) == 3
    validation_starts = [_times(panel, validation).min() for _, validation in folds]
    assert validation_starts == list(pd.date_range('2023-01-01', '2023-03-01', freq='MS', tz='UTC'))
    assert _times(panel, folds[0][0]).min() == pd.Timestamp('2022-01-01', tz='UTC')
    assert len(folds[1][0]) > len(folds[0][0])
    assert len(folds[2][0]) > len(folds[1][0])


def test_symbols_at_the_same_hour_always_stay_on_the_same_side(panel):
    for train, validation in walk_forward_splits(panel, target_column='target'):
        train_rows = panel.iloc[train]
        validation_rows = panel.iloc[validation]

        assert train_rows.groupby('open_time')['symbol'].nunique().eq(2).all()
        assert validation_rows.groupby('open_time')['symbol'].nunique().eq(2).all()
        assert set(train_rows['open_time']).isdisjoint(validation_rows['open_time'])


def test_training_respects_purge_embargo_and_final_holdout(panel):
    horizon = pd.Timedelta(hours=4)
    holdout_start = pd.Timestamp('2023-04-01', tz='UTC')

    for train, validation in walk_forward_splits(panel, horizon=horizon, target_column='target'):
        train_times = _times(panel, train)
        validation_times = _times(panel, validation)
        validation_start = validation_times.min()

        assert train_times.max() < validation_start - horizon
        assert (train_times + horizon < validation_start).all()
        assert (validation_times + horizon < holdout_start).all()


def test_horizon_controls_purge_and_validation_label_maturity(panel):
    horizon = pd.Timedelta(hours=24)
    folds = walk_forward_splits(panel, horizon=horizon, target_column='target')
    last_validation = _times(panel, folds[-1][1])

    assert (last_validation + horizon < pd.Timestamp('2023-04-01', tz='UTC')).all()
    assert last_validation.max() == pd.Timestamp('2023-03-30 23:00', tz='UTC')


def test_rows_with_missing_targets_are_excluded(panel):
    frame = panel.copy()
    missing_train = (frame['symbol'] == 'BTCUSDT') & (
        frame['open_time'] == pd.Timestamp('2022-12-01', tz='UTC')
    )
    missing_validation = (frame['symbol'] == 'ETHUSDT') & (
        frame['open_time'] == pd.Timestamp('2023-01-01', tz='UTC')
    )
    frame.loc[missing_train | missing_validation, 'target'] = float('nan')

    train, validation = walk_forward_splits(frame, target_column='target')[0]

    assert not missing_train.iloc[train].any()
    assert not missing_validation.iloc[validation].any()


def test_embargo_shorter_than_horizon_is_rejected(panel):
    with pytest.raises(ValueError, match='supérieur ou égal'):
        walk_forward_splits(
            panel,
            horizon=pd.Timedelta(hours=4),
            embargo=pd.Timedelta(hours=3),
            target_column='target',
        )


def test_insufficient_history_for_a_fold_is_rejected(panel):
    short = panel.loc[panel['open_time'] < pd.Timestamp('2023-03-01', tz='UTC')]

    with pytest.raises(ValueError, match='pas assez de données'):
        walk_forward_splits(short, target_column='target')


def test_duplicate_symbol_timestamp_is_rejected(panel):
    duplicate = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match='doublons'):
        walk_forward_splits(duplicate, target_column='target')


def test_holdout_uses_last_three_months_and_purges_training_labels(panel):
    train, holdout = holdout_split(panel, target_column='target')
    train_times = _times(panel, train)
    holdout_times = _times(panel, holdout)
    holdout_start = pd.Timestamp('2023-04-01', tz='UTC')

    assert train_times.max() == pd.Timestamp('2023-03-31 19:00', tz='UTC')
    assert (train_times + pd.Timedelta(hours=4) < holdout_start).all()
    assert holdout_times.min() == holdout_start
    assert holdout_times.max() == pd.Timestamp('2023-06-30 23:00', tz='UTC')
    assert set(train_times).isdisjoint(holdout_times)


def test_holdout_keeps_symbols_together_and_respects_the_horizon(panel):
    train, holdout = holdout_split(
        panel,
        horizon=pd.Timedelta(hours=24),
        target_column='target',
    )
    train_rows = panel.iloc[train]
    holdout_rows = panel.iloc[holdout]

    assert _times(panel, train).max() == pd.Timestamp('2023-03-30 23:00', tz='UTC')
    assert train_rows.groupby('open_time')['symbol'].nunique().eq(2).all()
    assert holdout_rows.groupby('open_time')['symbol'].nunique().eq(2).all()


def test_holdout_respects_embargo_longer_than_horizon(panel):
    train, _ = holdout_split(
        panel,
        horizon=pd.Timedelta(hours=4),
        embargo=pd.Timedelta(hours=12),
        target_column='target',
    )

    assert _times(panel, train).max() == pd.Timestamp('2023-03-31 11:00', tz='UTC')


def test_holdout_excludes_rows_with_missing_targets(panel):
    frame = panel.copy()
    missing_train = (frame['symbol'] == 'BTCUSDT') & (
        frame['open_time'] == pd.Timestamp('2023-03-01', tz='UTC')
    )
    missing_holdout = (frame['symbol'] == 'ETHUSDT') & (
        frame['open_time'] == pd.Timestamp('2023-04-01', tz='UTC')
    )
    frame.loc[missing_train | missing_holdout, 'target'] = float('nan')

    train, holdout = holdout_split(frame, target_column='target')

    assert not missing_train.iloc[train].any()
    assert not missing_holdout.iloc[holdout].any()


def test_holdout_rejects_embargo_shorter_than_horizon(panel):
    with pytest.raises(ValueError, match='supérieur ou égal'):
        holdout_split(
            panel,
            horizon=pd.Timedelta(hours=4),
            embargo=pd.Timedelta(hours=3),
            target_column='target',
        )


def test_holdout_rejects_empty_training_or_test_set(panel):
    recent = panel.loc[panel['open_time'] >= pd.Timestamp('2023-04-01', tz='UTC')]
    with pytest.raises(ValueError, match='train vide'):
        holdout_split(recent, target_column='target')

    missing_holdout_targets = panel.copy()
    missing_holdout_targets.loc[
        missing_holdout_targets['open_time'] >= pd.Timestamp('2023-04-01', tz='UTC'),
        'target',
    ] = float('nan')
    with pytest.raises(ValueError, match='holdout vide'):
        holdout_split(missing_holdout_targets, target_column='target')
