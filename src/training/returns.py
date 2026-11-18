"""Modèle `rendement` : prévision du rendement logarithmique futur à 4 h, évaluée en IC."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS
from training.dataset import ReturnTrainingDataset
from training.metrics import icir, information_coefficient
from training.splits import holdout_split, walk_forward_splits

MODEL_NAME = 'rendement'
TARGET_COLUMN = 'target_return'
HORIZON_HOURS = 4

Baseline = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def _zero(train: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    return np.zeros(len(rows), dtype='float64')


def _momentum_1h(train: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    return rows['return_1h'].to_numpy(dtype='float64')


def _momentum_4h(train: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    return rows['return_4h'].to_numpy(dtype='float64')


BASELINES: Mapping[str, Baseline] = {
    'zero': _zero,
    'momentum_1h': _momentum_1h,
    'momentum_4h': _momentum_4h,
}


def _candidates() -> dict[str, RegressorMixin]:
    return {
        'ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        'hgb': HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0
        ),
    }


class ReturnModel:
    """Régresseur entraîné sur le rendement logarithmique à 4 h."""

    def __init__(self, estimator: RegressorMixin) -> None:
        self.estimator = estimator

    def fit(self, rows: pd.DataFrame) -> ReturnModel:
        self.estimator.fit(rows.loc[:, list(FEATURE_COLUMNS)], rows[TARGET_COLUMN])
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.estimator.predict(rows.loc[:, list(FEATURE_COLUMNS)]),
            dtype='float64',
        )


def _prediction_frame(rows: pd.DataFrame, predicted: np.ndarray) -> pd.DataFrame:
    values = np.asarray(predicted, dtype='float64')
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError('le prédicteur doit renvoyer un rendement fini par ligne')
    return rows.loc[:, ['symbol', 'open_time', TARGET_COLUMN]].assign(prediction=values)


def score(predicted: np.ndarray, rows: pd.DataFrame) -> dict[str, float]:
    """IC moyen et ICIR transversal des prédictions pour un jeu de lignes."""
    coefficients = information_coefficient(_prediction_frame(rows, predicted))
    finite = coefficients[np.isfinite(coefficients.to_numpy(dtype='float64'))]
    mean_ic = float(finite.mean()) if not finite.empty else float('nan')
    return {'ic_mean': mean_ic, 'icir': icir(coefficients)}


@dataclass
class Evaluation:
    """Scores IC/ICIR par fold pour les candidats appris et les références."""

    folds: list[dict[str, dict[str, float]]] = field(default_factory=list)

    def mean(self, predictor: str, metric: str) -> float:
        values = np.asarray([fold[predictor][metric] for fold in self.folds], dtype='float64')
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else float('nan')

    def predictors(self) -> list[str]:
        return list(self.folds[0]) if self.folds else []


def cross_validate(frame: pd.DataFrame) -> Evaluation:
    """Évalue candidats et références sur les folds walk-forward, sans consulter le holdout."""
    splits = walk_forward_splits(
        frame,
        horizon=timedelta(hours=HORIZON_HOURS),
        target_column=TARGET_COLUMN,
    )
    evaluation = Evaluation()
    for train_index, validation_index in splits:
        train, validation = frame.iloc[train_index], frame.iloc[validation_index]
        fold = {name: score(baseline(train, validation), validation) for name, baseline in BASELINES.items()}
        for name, estimator in _candidates().items():
            model = ReturnModel(estimator).fit(train)
            fold[name] = score(model.predict(validation), validation)
        evaluation.folds.append(fold)
    return evaluation


@dataclass(frozen=True)
class TrainingResult:
    candidate: str
    model: ReturnModel
    evaluation: Evaluation
    holdout: dict[str, dict[str, float]]
    dataset: ReturnTrainingDataset
    forecasts: pd.DataFrame

    def beats_baselines(self) -> bool:
        """Le candidat doit battre au moins la meilleure référence ayant un IC défini."""
        baseline_scores = [
            self.holdout[name]['ic_mean'] for name in BASELINES if np.isfinite(self.holdout[name]['ic_mean'])
        ]
        candidate_score = self.holdout[self.candidate]['ic_mean']
        return bool(
            baseline_scores and np.isfinite(candidate_score) and candidate_score > max(baseline_scores)
        )


def fit_return_model(dataset: ReturnTrainingDataset) -> TrainingResult:
    """Sélectionne par IC walk-forward, réentraîne hors holdout et score le test final une fois."""
    frame = dataset.frame
    evaluation = cross_validate(frame)
    learned = [name for name in evaluation.predictors() if name not in BASELINES]
    candidate_scores = {name: evaluation.mean(name, 'ic_mean') for name in learned}
    defined_scores = {name: value for name, value in candidate_scores.items() if np.isfinite(value)}
    if not defined_scores:
        raise ValueError('aucun candidat ne produit un IC défini sur les folds de validation')
    candidate = max(defined_scores, key=defined_scores.__getitem__)

    train_index, holdout_index = holdout_split(
        frame,
        horizon=timedelta(hours=HORIZON_HOURS),
        target_column=TARGET_COLUMN,
    )
    train, holdout = frame.iloc[train_index], frame.iloc[holdout_index]
    model = ReturnModel(_candidates()[candidate]).fit(train)

    forecasts = holdout.loc[:, ['symbol', 'open_time', TARGET_COLUMN]].reset_index(drop=True)
    for name, baseline in BASELINES.items():
        forecasts[name] = baseline(train, holdout)
    forecasts[candidate] = model.predict(holdout)
    holdout_scores = {
        name: score(forecasts[name].to_numpy(dtype='float64'), holdout) for name in (*BASELINES, candidate)
    }
    return TrainingResult(
        candidate=candidate,
        model=model,
        evaluation=evaluation,
        holdout=holdout_scores,
        dataset=dataset,
        forecasts=forecasts,
    )
