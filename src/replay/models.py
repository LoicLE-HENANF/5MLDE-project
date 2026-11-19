"""Ce qui distingue les deux modèles dans le rejeu : entraînement, critères, prévisions, retours, santé.

Tout est délégué au code de production (`training`, `serving`) ; seul le registre change de nom
(`<modèle>-replay`) pour que le champion live ne soit jamais touché.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from replay import batch

REPLAY_SUFFIX = '-replay'


@dataclass(frozen=True)
class ModelSpec:
    name: str
    horizon_hours: int
    metric: str
    fit: Callable[[pd.DataFrame, pd.Timestamp], Any]
    log: Callable[[Any, Any], tuple[str, Any]]
    level1: Callable[[Any], bool]
    level2: Callable[[Any], bool]
    score: Callable[[Any], float]
    holdout: Callable[[Any], pd.DataFrame]
    champion_score: Callable[[np.ndarray, pd.DataFrame], float]
    rows: Callable[..., pd.DataFrame]
    feedback: Callable[..., pd.DataFrame]
    health: Callable[..., list[dict[str, Any]]]

    @property
    def replay_name(self) -> str:
        return self.name + REPLAY_SUFFIX


def _volatility() -> ModelSpec:
    from serving import health_rows, realized_feedback
    from training import volatility
    from training.dataset import prepare_volatility_dataset
    from training.metrics import qlike

    def fit(candles: pd.DataFrame, as_of: pd.Timestamp):
        dataset = prepare_volatility_dataset(candles, hours=volatility.HORIZON_HOURS, as_of=as_of)
        return volatility.fit_volatility_model(dataset)

    def champion_score(predicted: np.ndarray, holdout: pd.DataFrame) -> float:
        return qlike(predicted, holdout[volatility.TARGET_COLUMN].to_numpy(dtype='float64'))

    return ModelSpec(
        name=volatility.MODEL_NAME,
        horizon_hours=volatility.HORIZON_HOURS,
        metric='qlike',
        fit=fit,
        log=lambda result, connector: volatility.log_training(result, connector, run_name='rejeu-volatilite'),
        level1=lambda result: result.beats_baselines(),
        level2=lambda result: result.beats_persistence_strategy(),
        score=lambda result: result.holdout[result.candidate]['qlike'],
        holdout=lambda result: volatility.holdout_frame(result.dataset),
        champion_score=champion_score,
        rows=lambda features, predicted, start, end, *, with_positions: batch.volatility_rows(
            features, predicted, start, end, with_positions=with_positions
        ),
        feedback=realized_feedback,
        health=health_rows,
    )


def _returns() -> ModelSpec:
    from serving import realized_return_feedback, return_health_rows
    from training import returns
    from training.dataset import prepare_return_dataset

    def fit(candles: pd.DataFrame, as_of: pd.Timestamp):
        return returns.fit_return_model(prepare_return_dataset(candles, as_of=as_of))

    def champion_score(predicted: np.ndarray, holdout: pd.DataFrame) -> float:
        ic = returns.score(predicted, holdout)['ic_mean']
        if not np.isfinite(ic):
            raise ValueError('IC du champion indéfini sur le test final : comparaison impossible')
        return -ic

    return ModelSpec(
        name=returns.MODEL_NAME,
        horizon_hours=returns.HORIZON_HOURS,
        metric='ic_spearman',
        fit=fit,
        log=lambda result, connector: returns.log_training(result, connector, run_name='rejeu-rendement'),
        level1=lambda result: result.beats_baselines(),
        level2=lambda result: result.beats_long_short_strategy(),
        score=lambda result: -result.holdout[result.candidate]['ic_mean'],
        holdout=lambda result: returns.holdout_frame(result.dataset),
        champion_score=champion_score,
        rows=lambda features, predicted, start, end, *, with_positions: batch.return_rows(
            features,
            predicted,
            start,
            end,
            horizon_hours=returns.HORIZON_HOURS,
            with_positions=with_positions,
        ),
        feedback=realized_return_feedback,
        health=return_health_rows,
    )


SPECS: dict[str, Callable[[], ModelSpec]] = {'volatilite': _volatility, 'rendement': _returns}


def spec(name: str) -> ModelSpec:
    if name not in SPECS:
        raise ValueError(f'modèle {name!r} inconnu (attendu : {sorted(SPECS)})')
    return SPECS[name]()
