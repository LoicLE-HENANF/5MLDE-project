"""Graphiques du modèle `rendement`, joints à son run MLflow."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from training.metrics import information_coefficient

if TYPE_CHECKING:
    from training.returns import TrainingResult

REFERENCE_STYLE = {'linewidth': 1.0, 'alpha': 0.7}
CANDIDATE_STYLE = {'linewidth': 2.2}


def build(result: TrainingResult) -> dict[str, Figure]:
    """Construit les graphiques disponibles, nommés sans extension."""
    builders: dict[str, Callable[[TrainingResult], Figure | None]] = {
        'ic_par_fold': ic_by_fold,
        'ic_cumule_test_final': cumulative_ic,
        'ic_test_final': holdout_ic,
        'strategies_valeur_portefeuille': equity_curves,
        'courbe_apprentissage': learning_curve,
        'importance_features': importance,
    }
    figures = {}
    for name, builder in builders.items():
        figure = builder(result)
        if figure is not None:
            figures[name] = figure
    return figures


def close_all() -> None:
    plt.close('all')


def _style(predictor: str, candidate: str) -> dict:
    return CANDIDATE_STYLE if predictor == candidate else REFERENCE_STYLE


def ic_by_fold(result: TrainingResult) -> Figure | None:
    if not result.evaluation.folds:
        return None
    figure, axis = plt.subplots(figsize=(10, 4.5))
    folds = np.arange(1, len(result.evaluation.folds) + 1)
    for predictor in result.evaluation.predictors():
        values = [fold[predictor]['ic_mean'] for fold in result.evaluation.folds]
        axis.plot(folds, values, marker='o', label=predictor, **_style(predictor, result.candidate))
    axis.axhline(0, color='black', linewidth=0.7)
    axis.set(
        title=f'IC de Spearman par fold walk-forward (candidat : {result.candidate})',
        xlabel='fold (mois de validation)',
        ylabel='IC moyen (plus haut = mieux)',
        xticks=folds,
    )
    axis.legend(ncols=3)
    figure.tight_layout()
    return figure


def cumulative_ic(result: TrainingResult) -> Figure | None:
    table = result.forecasts
    if table.empty:
        return None
    predictors = [name for name in ('momentum_4h', result.candidate) if name in table]
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for predictor in predictors:
        rows = table.loc[:, ['symbol', 'open_time', 'target_return']].assign(
            prediction=table[predictor].to_numpy(dtype='float64')
        )
        coefficients = information_coefficient(rows)
        valid = coefficients.dropna()
        if not valid.empty:
            axis.plot(
                valid.index,
                valid.expanding().mean(),
                label=predictor,
                **_style(predictor, result.candidate),
            )
    if not axis.lines:
        plt.close(figure)
        return None
    axis.axhline(0, color='black', linewidth=0.7)
    axis.set(title='IC moyen cumulatif sur le test final', xlabel='date', ylabel='IC moyen')
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def holdout_ic(result: TrainingResult) -> Figure | None:
    metrics = {
        predictor: values['ic_mean']
        for predictor, values in result.holdout.items()
        if np.isfinite(values['ic_mean'])
    }
    if not metrics:
        return None
    names = sorted(metrics, key=metrics.__getitem__)
    colors = ['tab:green' if name == result.candidate else 'tab:grey' for name in names]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.barh(names, [metrics[name] for name in names], color=colors)
    axis.bar_label(bars, fmt='%.4f', padding=3)
    axis.axvline(0, color='black', linewidth=0.7)
    axis.invert_yaxis()
    axis.set(title='IC moyen sur le test final', xlabel='IC de Spearman (plus haut = mieux)')
    figure.tight_layout()
    return figure


def equity_curves(result: TrainingResult) -> Figure | None:
    if not result.equity:
        return None
    figure, axis = plt.subplots(figsize=(11, 4.5))
    for name, equity in result.equity.items():
        sharpe = result.strategy.get(name, {}).get('sharpe', float('nan'))
        predictor = result.candidate if name.startswith(f'{result.candidate}_') else name
        axis.plot(
            equity.index,
            equity.to_numpy(),
            label=f'{name} (Sharpe {sharpe:.2f})',
            **_style(predictor, result.candidate),
        )
    axis.axhline(1.0, color='black', linewidth=0.6)
    axis.set(
        title='Stratégies sur le test final (frais inclus)',
        xlabel='date',
        ylabel='valeur du portefeuille (base 1)',
    )
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def learning_curve(result: TrainingResult) -> Figure | None:
    curve = result.learning_curve
    if curve is None or curve.empty:
        return None
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(curve['iteration'], curve['train'], label='entraînement')
    axis.plot(curve['iteration'], curve['validation'], label='validation')
    axis.set(
        title="Courbe d'apprentissage HistGradientBoosting (dernier fold)",
        xlabel='itération',
        ylabel='erreur quadratique du rendement',
    )
    axis.legend()
    figure.tight_layout()
    return figure


def importance(result: TrainingResult) -> Figure | None:
    values = result.importance
    if values is None or values.empty:
        return None
    top = values.head(20).iloc[::-1]
    figure, axis = plt.subplots(figsize=(9, 0.3 * len(top) + 1.5))
    axis.barh(top.index, top.to_numpy(), color=np.where(top.to_numpy() > 0, 'tab:blue', 'tab:red'))
    axis.set(
        title=f'Importance par permutation ({result.candidate}, test final)',
        xlabel="hausse de l'erreur quadratique quand la feature est mélangée",
    )
    figure.tight_layout()
    return figure
