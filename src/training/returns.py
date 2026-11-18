"""Modèle `rendement` : prévision du rendement logarithmique futur à 4 h, évaluée en IC."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
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
from strategies import DEFAULT_FEE, BacktestResult, backtest, long_cash, long_short, to_wide
from training import return_figures
from training.dataset import ReturnTrainingDataset, prepare_return_dataset
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


class RendementPyfunc(mlflow.pyfunc.PythonModel):
    """Modèle servi : features en entrée, rendement logarithmique à 4 h en sortie."""

    def __init__(self, estimator: RegressorMixin) -> None:
        self.estimator = estimator

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        return np.asarray(
            self.estimator.predict(model_input.loc[:, list(FEATURE_COLUMNS)]),
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
    strategy: dict[str, dict[str, float]] = field(default_factory=dict)
    equity: dict[str, pd.Series] = field(default_factory=dict)
    learning_curve: pd.DataFrame | None = None
    importance: pd.Series | None = None

    def beats_baselines(self) -> bool:
        """Le candidat doit battre au moins la meilleure référence ayant un IC défini."""
        baseline_scores = [
            self.holdout[name]['ic_mean'] for name in BASELINES if np.isfinite(self.holdout[name]['ic_mean'])
        ]
        candidate_score = self.holdout[self.candidate]['ic_mean']
        return bool(
            baseline_scores and np.isfinite(candidate_score) and candidate_score > max(baseline_scores)
        )

    def beats_long_short_strategy(self) -> bool:
        candidate = self.strategy.get(f'{self.candidate}_long_short', {}).get('sharpe', float('nan'))
        references = [
            self.strategy.get(name, {}).get('sharpe', float('nan'))
            for name in ('brut', 'momentum_1h_long_short', 'momentum_4h_long_short')
        ]
        return bool(
            np.isfinite(candidate)
            and all(np.isfinite(reference) for reference in references)
            and all(candidate > reference for reference in references)
        )

    def beats_long_cash_strategy(self) -> bool:
        candidate = self.strategy.get(f'{self.candidate}_long_cash', {}).get('sharpe', float('nan'))
        references = [
            self.strategy.get(name, {}).get('sharpe', float('nan'))
            for name in ('brut', 'momentum_1h_long_cash', 'momentum_4h_long_cash')
        ]
        return bool(
            np.isfinite(candidate)
            and all(np.isfinite(reference) for reference in references)
            and all(candidate > reference for reference in references)
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
    backtests = strategy_backtests(
        holdout,
        {name: forecasts[name].to_numpy(dtype='float64') for name in (*BASELINES, candidate)},
        candidate=candidate,
    )
    return TrainingResult(
        candidate=candidate,
        model=model,
        evaluation=evaluation,
        holdout=holdout_scores,
        dataset=dataset,
        forecasts=forecasts,
        strategy={name: result.metrics for name, result in backtests.items()},
        equity={name: result.frame['equity'] for name, result in backtests.items()},
        learning_curve=learning_curve(frame),
        importance=feature_importance(model, holdout),
    )


def strategy_backtests(
    holdout: pd.DataFrame,
    forecasts: Mapping[str, np.ndarray],
    *,
    candidate: str,
    fee: float = DEFAULT_FEE,
) -> dict[str, BacktestResult]:
    """Backteste long/short et long/cash, face à l'équipondéré brut et au momentum."""
    identifiers = holdout.loc[:, ['symbol', 'open_time']].reset_index(drop=True)
    presence = to_wide(identifiers.assign(prediction=1.0)).notna()
    raw = presence.astype('float64').div(presence.sum(axis=1), axis=0).fillna(0.0)
    candles = _candles_from_returns(holdout)
    results = {'brut': backtest(raw, candles, fee=fee)}
    for predictor, values in forecasts.items():
        predictions = identifiers.assign(prediction=np.asarray(values, dtype='float64'))
        positions = {
            f'{predictor}_long_short': long_short(predictions),
            f'{predictor}_long_cash': long_cash(predictions, threshold=DEFAULT_FEE),
        }
        results.update({name: backtest(weights, candles, fee=fee) for name, weights in positions.items()})
    expected = {f'{candidate}_long_short', f'{candidate}_long_cash'}
    if not expected.issubset(results):
        raise ValueError(f'prévisions du candidat absentes : {sorted(expected - set(results))}')
    return results


def _candles_from_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Prix relatifs reconstruits depuis les rendements horaires historiques."""
    ordered = frame.loc[:, ['symbol', 'open_time', 'return_1h']].sort_values(['symbol', 'open_time'])
    close = np.exp(ordered.groupby('symbol')['return_1h'].cumsum())
    return ordered.loc[:, ['symbol', 'open_time']].assign(close=close.to_numpy()).reset_index(drop=True)


def learning_curve(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Erreurs quadratiques train/validation de HGB par itération sur le dernier fold."""
    splits = walk_forward_splits(
        frame,
        horizon=timedelta(hours=HORIZON_HOURS),
        target_column=TARGET_COLUMN,
    )
    if not splits:
        return None
    train_index, validation_index = splits[-1]
    train, validation = frame.iloc[train_index], frame.iloc[validation_index]
    estimator = _candidates()['hgb']
    columns = list(FEATURE_COLUMNS)
    train_target = train[TARGET_COLUMN].to_numpy(dtype='float64')
    validation_target = validation[TARGET_COLUMN].to_numpy(dtype='float64')
    estimator.fit(train.loc[:, columns], train_target)
    rows = []
    stages = zip(
        estimator.staged_predict(train.loc[:, columns]),
        estimator.staged_predict(validation.loc[:, columns]),
        strict=True,
    )
    for iteration, (train_predicted, validation_predicted) in enumerate(stages, start=1):
        rows.append(
            {
                'iteration': iteration,
                'train': float(np.mean((train_predicted - train_target) ** 2)),
                'validation': float(np.mean((validation_predicted - validation_target) ** 2)),
            }
        )
    return pd.DataFrame(rows)


def feature_importance(model: ReturnModel, holdout: pd.DataFrame, *, repeats: int = 3) -> pd.Series:
    """Importance par permutation sur le holdout, à titre descriptif."""
    result = permutation_importance(
        model.estimator,
        holdout.loc[:, list(FEATURE_COLUMNS)],
        holdout[TARGET_COLUMN].to_numpy(dtype='float64'),
        scoring='neg_mean_squared_error',
        n_repeats=repeats,
        random_state=0,
    )
    return pd.Series(result.importances_mean, index=list(FEATURE_COLUMNS)).sort_values(ascending=False)


def log_training(
    result: TrainingResult,
    connector,
    *,
    run_name: str = 'train-rendement',
) -> tuple[str, Any]:
    """Logge les paramètres, métriques, graphiques et modèle pyfunc dans MLflow."""
    dataset = result.dataset
    tags = {**dataset.identity.tags(), 'modele': MODEL_NAME, 'candidat': result.candidate}
    with connector.start_run(run_name, tags=tags) as run:
        mlflow.log_params(
            {
                'cible': TARGET_COLUMN,
                'horizon_heures': HORIZON_HOURS,
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
                    if np.isfinite(value):
                        mlflow.log_metric(f'cv_{metric}_{predictor}', value, step=step)
        for predictor in result.evaluation.predictors():
            for metric in ('ic_mean', 'icir'):
                value = result.evaluation.mean(predictor, metric)
                if np.isfinite(value):
                    mlflow.log_metric(f'cv_mean_{metric}_{predictor}', value)
        for predictor, metrics in result.holdout.items():
            for metric, value in metrics.items():
                if np.isfinite(value):
                    mlflow.log_metric(f'test_{metric}_{predictor}', value)
        mlflow.log_metric('test_bat_references', float(result.beats_baselines()))
        mlflow.log_metric('strategie_bat_long_short', float(result.beats_long_short_strategy()))
        mlflow.log_metric('strategie_bat_long_cash', float(result.beats_long_cash_strategy()))
        for strategy, metrics in result.strategy.items():
            for metric, value in metrics.items():
                if np.isfinite(value):
                    mlflow.log_metric(f'strategie_{metric}_{strategy}', value)
        mlflow.log_text(json.dumps(list(FEATURE_COLUMNS), indent=2), 'features.json')
        mlflow.log_text(dataset.quality_report.summary(), 'qualite_donnees.txt')
        try:
            for name, figure in return_figures.build(result).items():
                mlflow.log_figure(figure, f'graphiques/{name}.png')
        finally:
            return_figures.close_all()

        sample = dataset.frame.loc[:, list(FEATURE_COLUMNS)].head(5)
        info = connector.log_model(
            mlflow.pyfunc.log_model,
            name=MODEL_NAME,
            python_model=RendementPyfunc(result.model.estimator),
            input_example=sample,
            tags=dataset.identity.tags(),
        )
        return run.info.run_id, info


def holdout_frame(dataset: ReturnTrainingDataset) -> pd.DataFrame:
    """Lignes du test final selon les règles de purge et d'embargo."""
    _, holdout_index = holdout_split(
        dataset.frame,
        horizon=timedelta(hours=HORIZON_HOURS),
        target_column=TARGET_COLUMN,
    )
    return dataset.frame.iloc[holdout_index]


def champion_scores(connector, holdout: pd.DataFrame) -> tuple[float | None, float | None]:
    """IC négatif et Sharpe long/short du champion sur le holdout du candidat."""
    from registry import CHAMPION

    if connector.get_version(MODEL_NAME, CHAMPION) is None:
        return None, None
    champion = connector.load_model(MODEL_NAME, CHAMPION)
    predicted = np.asarray(
        champion.predict(holdout.loc[:, list(FEATURE_COLUMNS)]),
        dtype='float64',
    )
    champion_ic = score(predicted, holdout)['ic_mean']
    if not np.isfinite(champion_ic):
        raise ValueError('IC du champion indéfini sur le holdout : comparaison de promotion impossible')
    strategy = strategy_backtests(
        holdout,
        {'champion': predicted},
        candidate='champion',
    )
    return -champion_ic, strategy['champion_long_short'].metrics['sharpe']


def promote(
    result: TrainingResult,
    connector,
    info,
    *,
    record=None,
    allow_unreleased: bool = False,
):
    """Applique les critères IC et long/short puis publie dans le registre."""
    from registry import decide, publish

    champion_ic_score, champion_sharpe = champion_scores(connector, holdout_frame(result.dataset))
    level2 = result.beats_long_short_strategy()
    if champion_sharpe is not None:
        level2 = level2 and (result.strategy[f'{result.candidate}_long_short']['sharpe'] >= champion_sharpe)
    decision = decide(
        level1=result.beats_baselines(),
        level2=level2,
        score=-result.holdout[result.candidate]['ic_mean'],
        champion_score=champion_ic_score,
    )
    version = publish(connector, info, MODEL_NAME, decision, record=record, allow_unreleased=allow_unreleased)
    return decision, version


def main(argv: list[str] | None = None) -> int:
    from connectors.config import MlflowConfig, PostgresConfig
    from connectors.mlflow import MlflowConnector
    from connectors.postgres import PostgresConnector
    from training.volatility import load_candles

    parser = argparse.ArgumentParser(description='Entraîne et logge le modèle de rendement.')
    parser.add_argument(
        '--as-of',
        type=datetime.fromisoformat,
        default=None,
        help='instant de référence (UTC)',
    )
    parser.add_argument(
        '--registre',
        action='store_true',
        help='enregistre le modèle et applique la règle de promotion',
    )
    parser.add_argument(
        '--non-publie',
        action='store_true',
        help="autorise l'enregistrement de code non publié (essais uniquement)",
    )
    args = parser.parse_args(argv)

    dataset = prepare_return_dataset(load_candles(), as_of=args.as_of)
    result = fit_return_model(dataset)
    connector = MlflowConnector(MlflowConfig.from_env())
    run_id, info = log_training(result, connector)
    print(f'candidat : {result.candidate}')
    for predictor, metrics in result.holdout.items():
        print(f'  {predictor:<12} IC {metrics["ic_mean"]:.4f}  ICIR {metrics["icir"]:.4f}')
    print(f'bat les références : {result.beats_baselines()}')
    for strategy, metrics in result.strategy.items():
        print(
            f'  stratégie {strategy:<28} Sharpe {metrics["sharpe"]:.2f}'
            f'  rendement {metrics["total_return"]:.2%}  frais {metrics["fees"]:.2%}'
        )
    print(connector.run_url(run_id))
    if args.registre:
        with PostgresConnector(PostgresConfig.from_env()) as db:
            decision, version = promote(
                result,
                connector,
                info,
                record=db.record_model_event,
                allow_unreleased=args.non_publie,
            )
        print(f'registre : {decision.summary()}')
        if version is not None:
            print(f'  {MODEL_NAME} version {version.version} -> {decision.alias}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
