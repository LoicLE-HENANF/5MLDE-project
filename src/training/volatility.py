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

Évaluation à deux niveaux sur le test final :
- niveau 1 (statistique) : QLIKE du candidat inférieur à celui de la meilleure référence ;
- niveau 2 (stratégie) : un portefeuille équipondéré dont l'exposition est ciblée par la
  prévision (backtest frais inclus) a un Sharpe au moins égal au même ciblage par la persistance.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS
from strategies import DEFAULT_FEE, BacktestResult, backtest, scale_by_volatility, to_wide
from training import figures
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
    strategy: dict[str, dict[str, float]] = field(default_factory=dict)
    # Données des graphiques du run (facultatives : absentes, le graphique correspondant est omis).
    forecasts: pd.DataFrame | None = (
        None  # test final : symbol, open_time, realise, une colonne par prédicteur
    )
    equity: dict[str, pd.Series] = field(default_factory=dict)  # valeur du portefeuille par stratégie
    learning_curve: pd.DataFrame | None = None  # hgb par itération : train, validation, qlike_validation
    importance: pd.Series | None = None  # importance par permutation sur le test final

    def beats_baselines(self) -> bool:
        best_baseline = min(self.holdout[name]['qlike'] for name in BASELINES)
        return self.holdout[self.candidate]['qlike'] < best_baseline

    def beats_persistence_strategy(self) -> bool:
        """Niveau 2 : le ciblage de volatilité piloté par le modèle fait au moins aussi bien
        (Sharpe sur le test final) que le même ciblage piloté par la persistance."""
        return self.strategy[self.candidate]['sharpe'] >= self.strategy['persistance']['sharpe']


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
    predicted = model.predict(holdout)
    scores[candidate] = score(predicted, realized)
    forecasts = {'persistance': _persistence(train, holdout), candidate: predicted}
    target = float(train[TARGET_COLUMN].median())
    backtests = strategy_backtests(holdout, forecasts, target=target)
    return TrainingResult(
        candidate,
        model,
        evaluation,
        scores,
        dataset,
        {name: result.metrics for name, result in backtests.items()},
        forecasts=_holdout_forecasts(train, holdout, {candidate: predicted}),
        equity={name: result.frame['equity'] for name, result in backtests.items()},
        learning_curve=learning_curve(frame, hours=dataset.hours),
        importance=feature_importance(model, holdout),
    )


def _holdout_forecasts(
    train: pd.DataFrame, holdout: pd.DataFrame, learned: Mapping[str, np.ndarray]
) -> pd.DataFrame:
    """Prévisions de chaque prédicteur sur le test final, à côté de la volatilité réalisée."""
    table = holdout.loc[:, ['symbol', 'open_time']].reset_index(drop=True)
    table['realise'] = holdout[TARGET_COLUMN].to_numpy(dtype='float64')
    for name, baseline in BASELINES.items():
        table[name] = np.asarray(baseline(train, holdout), dtype='float64')
    for name, values in learned.items():
        table[name] = np.asarray(values, dtype='float64')
    return table


def learning_curve(frame: pd.DataFrame, *, hours: int = HORIZON_HOURS) -> pd.DataFrame | None:
    """Courbe d'apprentissage de `hgb` sur le dernier fold walk-forward (test final exclu).

    Par itération du boosting : perte du modèle (erreur quadratique sur le log de la volatilité)
    en entraînement et en validation, et QLIKE en validation.
    """
    splits = walk_forward_splits(frame, horizon=timedelta(hours=hours), target_column=TARGET_COLUMN)
    if not splits:
        return None
    train_index, validation_index = splits[-1]
    train, validation = frame.iloc[train_index], frame.iloc[validation_index]
    model = LogVolatilityModel(_candidates()['hgb']).fit(train)
    estimator = model.estimator
    columns = list(FEATURE_COLUMNS)
    train_log = np.log(train[TARGET_COLUMN].to_numpy(dtype='float64'))
    validation_log = np.log(validation[TARGET_COLUMN].to_numpy(dtype='float64'))
    rows = []
    stages = zip(
        estimator.staged_predict(train.loc[:, columns]),
        estimator.staged_predict(validation.loc[:, columns]),
        strict=True,
    )
    for iteration, (train_pred, validation_pred) in enumerate(stages, start=1):
        rows.append(
            {
                'iteration': iteration,
                'train': float(np.mean((train_pred - train_log) ** 2)),
                'validation': float(np.mean((validation_pred - validation_log) ** 2)),
                'qlike_validation': qlike(np.exp(validation_pred), np.exp(validation_log)),
            }
        )
    return pd.DataFrame(rows)


def feature_importance(model: LogVolatilityModel, holdout: pd.DataFrame, *, repeats: int = 3) -> pd.Series:
    """Hausse de l'erreur quadratique (log-volatilité) sur le test final quand une feature est mélangée."""
    importances = permutation_importance(
        model.estimator,
        holdout.loc[:, list(FEATURE_COLUMNS)],
        np.log(holdout[TARGET_COLUMN].to_numpy(dtype='float64')),
        scoring='neg_mean_squared_error',
        n_repeats=repeats,
        random_state=0,
    )
    return pd.Series(importances.importances_mean, index=list(FEATURE_COLUMNS)).sort_values(ascending=False)


def strategy_scores(
    holdout: pd.DataFrame, forecasts: Mapping[str, np.ndarray], *, target: float, fee: float = DEFAULT_FEE
) -> dict[str, dict[str, float]]:
    """Métriques de `strategy_backtests`."""
    backtests = strategy_backtests(holdout, forecasts, target=target, fee=fee)
    return {name: result.metrics for name, result in backtests.items()}


def strategy_backtests(
    holdout: pd.DataFrame, forecasts: Mapping[str, np.ndarray], *, target: float, fee: float = DEFAULT_FEE
) -> dict[str, BacktestResult]:
    """Niveau 2 : backtest sur le test final d'un portefeuille équipondéré acheteur dont
    l'exposition est ciblée par chaque prévision de volatilité (`target` / prévision, sans levier).

    `brut` est le même portefeuille sans ciblage. Les rendements horaires viennent de la feature
    `return_1h` de la bougie suivante (aucune donnée hors du jeu d'entraînement).
    """
    candles = _candles_from_returns(holdout)
    keys = holdout.loc[:, ['symbol', 'open_time']].reset_index(drop=True)
    present = to_wide(keys.assign(prediction=1.0)).notna()
    raw = present.astype('float64') / present.shape[1]
    results = {'brut': backtest(raw, candles, fee=fee)}
    for name, forecast in forecasts.items():
        volatility = to_wide(keys.assign(prediction=np.asarray(forecast, dtype='float64')))
        positions = scale_by_volatility(raw, volatility, target=target)
        results[name] = backtest(positions, candles, fee=fee)
    return results


def _candles_from_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Prix relatifs reconstruits par symbole : close(t) / close(t - 1 h) = exp(return_1h(t))."""
    ordered = frame.loc[:, ['symbol', 'open_time', 'return_1h']].sort_values(['symbol', 'open_time'])
    close = np.exp(ordered.groupby('symbol')['return_1h'].cumsum())
    return ordered.loc[:, ['symbol', 'open_time']].assign(close=close.to_numpy()).reset_index(drop=True)


def log_training(result: TrainingResult, connector, *, run_name: str = 'train-volatilite') -> tuple[str, Any]:
    """Logge le run MLflow (paramètres, métriques, modèle) ; renvoie (run_id, ModelInfo)."""
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
        for strategy, metrics in result.strategy.items():
            for metric, value in metrics.items():
                mlflow.log_metric(f'strategie_{metric}_{strategy}', value)
        if result.strategy:
            mlflow.log_metric('strategie_bat_persistance', float(result.beats_persistence_strategy()))
        mlflow.log_text(json.dumps(list(FEATURE_COLUMNS), indent=2), 'features.json')
        mlflow.log_text(result.dataset.quality_report.summary(), 'qualite_donnees.txt')
        for name, figure in figures.build(result).items():
            mlflow.log_figure(figure, f'graphiques/{name}.png')
        figures.close_all()

        sample = dataset.frame.loc[:, list(FEATURE_COLUMNS)].head(5)
        info = connector.log_model(
            mlflow.pyfunc.log_model,
            name=MODEL_NAME,
            python_model=VolatilityPyfunc(result.model.estimator),
            input_example=sample,
            tags=dataset.identity.tags(),
        )
        return run.info.run_id, info


def holdout_frame(dataset: VolatilityTrainingDataset) -> pd.DataFrame:
    """Lignes du test final (mêmes règles d'embargo que l'entraînement)."""
    _, holdout_index = holdout_split(
        dataset.frame, horizon=timedelta(hours=dataset.hours), target_column=TARGET_COLUMN
    )
    return dataset.frame.iloc[holdout_index]


def champion_score(connector, holdout: pd.DataFrame) -> float | None:
    """QLIKE du champion actuel rescoré sur le test final du candidat (None sans champion)."""
    from registry import CHAMPION

    if connector.get_version(MODEL_NAME, CHAMPION) is None:
        return None
    champion = connector.load_model(MODEL_NAME, CHAMPION)
    predicted = np.asarray(champion.predict(holdout.loc[:, list(FEATURE_COLUMNS)]), dtype='float64')
    return qlike(predicted, holdout[TARGET_COLUMN].to_numpy(dtype='float64'))


def promote(result: TrainingResult, connector, info, *, record=None, allow_unreleased: bool = False):
    """Applique la règle de promotion (niveaux 1 et 2, comparaison au champion) et publie."""
    from registry import decide, publish

    decision = decide(
        level1=result.beats_baselines(),
        level2=result.beats_persistence_strategy(),
        score=result.holdout[result.candidate]['qlike'],
        champion_score=champion_score(connector, holdout_frame(result.dataset)),
    )
    version = publish(connector, info, MODEL_NAME, decision, record=record, allow_unreleased=allow_unreleased)
    return decision, version


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
    parser.add_argument(
        '--registre', action='store_true', help='enregistre le modèle et applique la règle de promotion'
    )
    parser.add_argument(
        '--non-publie',
        action='store_true',
        help="autorise l'enregistrement de code non publié (essais uniquement)",
    )
    args = parser.parse_args(argv)

    dataset = prepare_volatility_dataset(load_candles(), hours=HORIZON_HOURS, as_of=args.as_of)
    result = fit_volatility_model(dataset)
    connector = MlflowConnector(MlflowConfig.from_env())
    run_id, info = log_training(result, connector)
    print(f'candidat : {result.candidate}')
    for predictor, metrics in result.holdout.items():
        print(f'  {predictor:<12} QLIKE {metrics["qlike"]:.4f}  MAE log-vol {metrics["mae_log_vol"]:.4f}')
    print(f'bat les références : {result.beats_baselines()}')
    for strategy, metrics in result.strategy.items():
        print(
            f'  stratégie {strategy:<12} Sharpe {metrics["sharpe"]:.2f}'
            f'  perte max {metrics["max_drawdown"]:.2%}'
        )
    print(connector.run_url(run_id))
    if args.registre:
        from connectors.config import PostgresConfig
        from connectors.postgres import PostgresConnector

        with PostgresConnector(PostgresConfig.from_env()) as db:
            decision, version = promote(
                result, connector, info, record=db.record_model_event, allow_unreleased=args.non_publie
            )
        print(f'registre : {decision.summary()}')
        if version is not None:
            print(f'  {MODEL_NAME} version {version.version} -> {decision.alias}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
