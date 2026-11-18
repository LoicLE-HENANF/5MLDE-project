"""Modèle `volatilite` : prévision de la volatilité réalisée à 24 h, évaluée en QLIKE.

Le modèle apprend le logarithme de la volatilité future à partir des features communes.
Il est comparé à trois références naïves sur les mêmes folds walk-forward puis sur le
test final (3 derniers mois, jamais vus pendant la sélection) :

- `persistance` : la volatilité réalisée des 24 dernières heures ;
- `semaine` : la volatilité réalisée des 168 dernières heures ramenée à 24 h ;
- `moyenne` : la volatilité moyenne (en log) du symbole sur l'entraînement.

Chaque candidat appris (Ridge, HistGradientBoosting) et chaque référence reçoit QLIKE et
MAE log-vol par fold. Le meilleur candidat (QLIKE moyen) est réentraîné sur toute la
période hors test final, évalué une seule fois sur le test final, puis loggé dans MLflow
avec l'identité des données et la provenance du code.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS
from training.dataset import VolatilityTrainingDataset, prepare_volatility_dataset
from training.metrics import log_volatility_mae, qlike
from training.splits import holdout_split, walk_forward_splits

MODEL_NAME = 'volatilite'
TARGET_COLUMN = 'target_volatility'
HORIZON_HOURS = 24

Baseline = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def _persistence(train: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    return rows['realized_vol_24h'].to_numpy(dtype='float64')


def _week(train: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    return rows['realized_vol_168h'].to_numpy(dtype='float64') * np.sqrt(HORIZON_HOURS / 168)


def _symbol_mean(train: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    means = np.log(train[TARGET_COLUMN]).groupby(train['symbol']).mean()
    overall = float(np.log(train[TARGET_COLUMN]).mean())
    return np.exp(rows['symbol'].map(means).fillna(overall).to_numpy(dtype='float64'))


BASELINES: Mapping[str, Baseline] = {
    'persistance': _persistence,
    'semaine': _week,
    'moyenne': _symbol_mean,
}


def _candidates() -> dict[str, RegressorMixin]:
    return {
        'ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        'hgb': HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0
        ),
    }


class LogVolatilityModel:
    """Régresseur entraîné sur log(volatilité) ; `predict` renvoie une volatilité positive."""

    def __init__(self, estimator: RegressorMixin) -> None:
        self.estimator = estimator

    def fit(self, rows: pd.DataFrame) -> LogVolatilityModel:
        self.estimator.fit(rows.loc[:, list(FEATURE_COLUMNS)], np.log(rows[TARGET_COLUMN]))
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.exp(self.estimator.predict(rows.loc[:, list(FEATURE_COLUMNS)]))


class VolatilityPyfunc(mlflow.pyfunc.PythonModel):
    """Modèle servi : features en entrée, volatilité à 24 h (échelle naturelle) en sortie."""

    def __init__(self, estimator: RegressorMixin) -> None:
        self.estimator = estimator

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        return np.exp(self.estimator.predict(model_input.loc[:, list(FEATURE_COLUMNS)]))


def score(predicted: np.ndarray, realized: pd.Series) -> dict[str, float]:
    values = realized.to_numpy(dtype='float64')
    return {'qlike': qlike(predicted, values), 'mae_log_vol': log_volatility_mae(predicted, values)}


@dataclass
class Evaluation:
    """Scores par fold de chaque prédicteur (candidats appris et références)."""

    folds: list[dict[str, dict[str, float]]] = field(default_factory=list)

    def mean(self, predictor: str, metric: str) -> float:
        return float(np.mean([fold[predictor][metric] for fold in self.folds]))

    def predictors(self) -> list[str]:
        return list(self.folds[0]) if self.folds else []


def cross_validate(frame: pd.DataFrame, *, hours: int = HORIZON_HOURS) -> Evaluation:
    """Évalue candidats et références sur les folds walk-forward (test final exclu)."""
    splits = walk_forward_splits(frame, horizon=timedelta(hours=hours), target_column=TARGET_COLUMN)
    if not splits:
        raise ValueError('aucun fold walk-forward : historique trop court')
    evaluation = Evaluation()
    for train_index, validation_index in splits:
        train, validation = frame.iloc[train_index], frame.iloc[validation_index]
        realized = validation[TARGET_COLUMN]
        fold = {name: score(baseline(train, validation), realized) for name, baseline in BASELINES.items()}
        for name, estimator in _candidates().items():
            fold[name] = score(LogVolatilityModel(estimator).fit(train).predict(validation), realized)
        evaluation.folds.append(fold)
    return evaluation


@dataclass(frozen=True)
class TrainingResult:
    candidate: str
    model: LogVolatilityModel
    evaluation: Evaluation
    holdout: dict[str, dict[str, float]]
    dataset: VolatilityTrainingDataset

    def beats_baselines(self) -> bool:
        best_baseline = min(self.holdout[name]['qlike'] for name in BASELINES)
        return self.holdout[self.candidate]['qlike'] < best_baseline


def fit_volatility_model(dataset: VolatilityTrainingDataset) -> TrainingResult:
    """Sélection par walk-forward, réentraînement hors test final, score unique sur le test final."""
    frame = dataset.frame
    evaluation = cross_validate(frame, hours=dataset.hours)
    learned = [name for name in evaluation.predictors() if name not in BASELINES]
    candidate = min(learned, key=lambda name: evaluation.mean(name, 'qlike'))

    train_index, holdout_index = holdout_split(
        frame, horizon=timedelta(hours=dataset.hours), target_column=TARGET_COLUMN
    )
    train, holdout = frame.iloc[train_index], frame.iloc[holdout_index]
    model = LogVolatilityModel(_candidates()[candidate]).fit(train)
    realized = holdout[TARGET_COLUMN]
    scores = {name: score(baseline(train, holdout), realized) for name, baseline in BASELINES.items()}
    scores[candidate] = score(model.predict(holdout), realized)
    return TrainingResult(candidate, model, evaluation, scores, dataset)


def log_training(result: TrainingResult, connector, *, run_name: str = 'train-volatilite') -> str:
    """Logge le run MLflow : paramètres, métriques par fold et test final, modèle sklearn."""
    dataset = result.dataset
    tags = {**dataset.identity.tags(), 'modele': MODEL_NAME, 'candidat': result.candidate}
    with connector.start_run(run_name, tags=tags) as run:
        mlflow.log_params(
            {
                'cible': TARGET_COLUMN,
                'horizon_heures': dataset.hours,
                'candidat': result.candidate,
                'features': len(FEATURE_COLUMNS),
                'lignes': len(dataset.frame),
                'folds': len(result.evaluation.folds),
                'as_of': dataset.as_of.isoformat(),
            }
        )
        for step, fold in enumerate(result.evaluation.folds):
            for predictor, metrics in fold.items():
                for metric, value in metrics.items():
                    mlflow.log_metric(f'cv_{metric}_{predictor}', value, step=step)
        for predictor in result.evaluation.predictors():
            for metric in ('qlike', 'mae_log_vol'):
                mlflow.log_metric(f'cv_mean_{metric}_{predictor}', result.evaluation.mean(predictor, metric))
        for predictor, metrics in result.holdout.items():
            for metric, value in metrics.items():
                mlflow.log_metric(f'test_{metric}_{predictor}', value)
        mlflow.log_metric('test_bat_references', float(result.beats_baselines()))
        mlflow.log_text(json.dumps(list(FEATURE_COLUMNS), indent=2), 'features.json')
        mlflow.log_text(result.dataset.quality_report.summary(), 'qualite_donnees.txt')

        sample = dataset.frame.loc[:, list(FEATURE_COLUMNS)].head(5)
        connector.log_model(
            mlflow.pyfunc.log_model,
            name=MODEL_NAME,
            python_model=VolatilityPyfunc(result.model.estimator),
            input_example=sample,
            tags=dataset.identity.tags(),
        )
        return run.info.run_id


def load_candles(env: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Historique Parquet complété des bougies récentes PostgreSQL (doublons écartés)."""
    from connectors.config import PostgresConfig
    from connectors.postgres import PostgresConnector
    from ingestion.store import CandleTable, ParquetStore, data_dir

    history = ParquetStore(data_dir(env)).read()
    with PostgresConnector(PostgresConfig.from_env(env)) as db:
        recent = CandleTable(db).read()
    candles = pd.concat([history, recent], ignore_index=True)
    candles = candles.drop_duplicates(['symbol', 'open_time'], keep='first')
    return candles.sort_values(['symbol', 'open_time'], ignore_index=True)


def main(argv: Sequence[str] | None = None) -> int:
    from connectors.config import MlflowConfig
    from connectors.mlflow import MlflowConnector

    parser = argparse.ArgumentParser(description='Entraîne et logge le modèle de volatilité.')
    parser.add_argument(
        '--as-of', type=datetime.fromisoformat, default=None, help='instant de référence (UTC)'
    )
    args = parser.parse_args(argv)

    dataset = prepare_volatility_dataset(load_candles(), hours=HORIZON_HOURS, as_of=args.as_of)
    result = fit_volatility_model(dataset)
    connector = MlflowConnector(MlflowConfig.from_env())
    run_id = log_training(result, connector)
    print(f'candidat : {result.candidate}')
    for predictor, metrics in result.holdout.items():
        print(f'  {predictor:<12} QLIKE {metrics["qlike"]:.4f}  MAE log-vol {metrics["mae_log_vol"]:.4f}')
    print(f'bat les références : {result.beats_baselines()}')
    print(connector.run_url(run_id))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
