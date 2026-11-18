"""Graphiques d'un run d'entraînement du modèle `volatilite`, joints au run MLflow (`graphiques/`).

Chaque graphique se lit seul : titre, axes nommés, candidat retenu mis en évidence. Un graphique
dont les données manquent dans le `TrainingResult` est simplement omis.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use('Agg')  # pas d'affichage : rendu en fichier uniquement (worker, CI)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from training.metrics import qlike

if TYPE_CHECKING:
    from training.volatility import TrainingResult

TOP_FEATURES = 20
REFERENCE_STYLE = {'linewidth': 1.0, 'alpha': 0.7}
CANDIDATE_STYLE = {'linewidth': 2.2}


def build(result: TrainingResult) -> dict[str, Figure]:
    """Graphiques disponibles pour ce résultat, par nom de fichier (sans extension)."""
    builders: dict[str, Callable[[TrainingResult], Figure | None]] = {
        'courbe_apprentissage': learning_curve,
        'qlike_par_fold': qlike_by_fold,
        'test_final_scores': holdout_scores,
        'test_final_qlike_hebdo': weekly_qlike,
        'test_final_prevision_vs_realise': forecast_vs_realized,
        'test_final_dispersion': scatter,
        'strategies_valeur_portefeuille': equity_curves,
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


def learning_curve(result: TrainingResult) -> Figure | None:
    """Perte de `hgb` par itération du boosting, entraînement et validation (dernier fold)."""
    curve = result.learning_curve
    if curve is None or curve.empty:
        return None
    figure, (loss_axis, qlike_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    loss_axis.plot(curve['iteration'], curve['train'], label='entraînement')
    loss_axis.plot(curve['iteration'], curve['validation'], label='validation')
    loss_axis.set(
        title='Perte (erreur quadratique sur log-volatilité)', xlabel='itération', ylabel='MSE log-vol'
    )
    loss_axis.legend()
    best = curve.loc[curve['qlike_validation'].idxmin()]
    qlike_axis.plot(curve['iteration'], curve['qlike_validation'], color='tab:orange')
    qlike_axis.axvline(best['iteration'], color='grey', linestyle='--', linewidth=1)
    qlike_axis.annotate(
        f'min {best["qlike_validation"]:.4f}\nitération {int(best["iteration"])}',
        (best['iteration'], best['qlike_validation']),
        textcoords='offset points',
        xytext=(8, 12),
    )
    qlike_axis.set(title='QLIKE en validation', xlabel='itération', ylabel='QLIKE')
    figure.suptitle("Courbe d'apprentissage hgb (dernier fold walk-forward)")
    figure.tight_layout()
    return figure


def qlike_by_fold(result: TrainingResult) -> Figure | None:
    """QLIKE de chaque prédicteur sur chaque fold walk-forward : stabilité dans le temps."""
    folds = result.evaluation.folds
    if not folds:
        return None
    figure, axis = plt.subplots(figsize=(10, 4.5))
    steps = np.arange(1, len(folds) + 1)
    for predictor in result.evaluation.predictors():
        values = [fold[predictor]['qlike'] for fold in folds]
        axis.plot(steps, values, marker='o', label=predictor, **_style(predictor, result.candidate))
    axis.set(
        title=f'QLIKE par fold walk-forward (candidat retenu : {result.candidate})',
        xlabel='fold (mois de validation)',
        ylabel='QLIKE (plus bas = mieux)',
        xticks=steps,
    )
    axis.legend(ncols=3)
    figure.tight_layout()
    return figure


def holdout_scores(result: TrainingResult) -> Figure | None:
    """QLIKE et MAE log-vol de chaque prédicteur sur le test final."""
    if not result.holdout:
        return None
    names = sorted(result.holdout, key=lambda name: result.holdout[name]['qlike'])
    colors = ['tab:green' if name == result.candidate else 'tab:grey' for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric, label in zip(axes, ('qlike', 'mae_log_vol'), ('QLIKE', 'MAE log-vol'), strict=True):
        values = [result.holdout[name][metric] for name in names]
        bars = axis.barh(names, values, color=colors)
        axis.bar_label(bars, fmt='%.4f', padding=3)
        axis.invert_yaxis()
        axis.set(title=f'{label} sur le test final', xlabel=f'{label} (plus bas = mieux)')
        axis.margins(x=0.2)
    figure.tight_layout()
    return figure


def weekly_qlike(result: TrainingResult) -> Figure | None:
    """QLIKE par semaine du test final : le candidat bat-il les références de façon régulière ?"""
    table = result.forecasts
    if table is None or table.empty:
        return None
    predictors = [name for name in table.columns if name not in ('symbol', 'open_time', 'realise')]
    week = table['open_time'].dt.tz_localize(None).dt.to_period('W').dt.start_time
    weekly = pd.DataFrame(
        {
            name: table.groupby(week).apply(lambda rows, name=name: qlike(rows[name], rows['realise']))
            for name in predictors
        }
    )
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for name in predictors:
        axis.plot(weekly.index, weekly[name], marker='.', label=name, **_style(name, result.candidate))
    axis.set(title='QLIKE hebdomadaire sur le test final', xlabel='semaine', ylabel='QLIKE')
    axis.legend(ncols=3)
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def forecast_vs_realized(result: TrainingResult) -> Figure | None:
    """Volatilité prévue (candidat, persistance) et réalisée, par symbole, sur le test final."""
    table = result.forecasts
    if table is None or table.empty or result.candidate not in table:
        return None
    symbols = sorted(table['symbol'].unique())
    figure, axes = plt.subplots(len(symbols), 1, figsize=(12, 2.4 * len(symbols)), sharex=True, squeeze=False)
    for axis, symbol in zip(axes[:, 0], symbols, strict=True):
        rows = table.loc[table['symbol'] == symbol].sort_values('open_time')
        axis.plot(rows['open_time'], rows['realise'], color='black', linewidth=0.8, label='réalisée')
        axis.plot(
            rows['open_time'], rows['persistance'], color='tab:grey', linewidth=0.8, label='persistance'
        )
        axis.plot(
            rows['open_time'],
            rows[result.candidate],
            color='tab:green',
            linewidth=1.2,
            label=result.candidate,
        )
        axis.set_yscale('log')
        axis.set_ylabel(symbol)
    axes[0, 0].legend(loc='upper right', ncols=3)
    axes[0, 0].set_title('Volatilité 24 h prévue et réalisée sur le test final (échelle log)')
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def scatter(result: TrainingResult) -> Figure | None:
    """Prévu contre réalisé (log-log) : un bon modèle suit la diagonale."""
    table = result.forecasts
    if table is None or table.empty or result.candidate not in table:
        return None
    predictors = ['persistance', result.candidate]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    realized = table['realise']
    low, high = realized.min(), realized.max()
    for axis, name in zip(axes, predictors, strict=True):
        axis.scatter(table[name], realized, s=3, alpha=0.25)
        axis.plot([low, high], [low, high], color='red', linewidth=1, label='prévision parfaite')
        correlation = np.corrcoef(np.log(table[name]), np.log(realized))[0, 1]
        axis.set(
            title=f'{name} (corrélation log {correlation:.2f})',
            xscale='log',
            yscale='log',
            xlabel='volatilité prévue',
            ylabel='volatilité réalisée',
        )
        axis.legend(loc='upper left')
    figure.suptitle('Prévu contre réalisé sur le test final')
    figure.tight_layout()
    return figure


def equity_curves(result: TrainingResult) -> Figure | None:
    """Valeur du portefeuille de chaque stratégie (frais inclus) sur le test final."""
    if not result.equity:
        return None
    figure, axis = plt.subplots(figsize=(11, 4.5))
    for name, equity in result.equity.items():
        sharpe = result.strategy.get(name, {}).get('sharpe', float('nan'))
        axis.plot(
            equity.index,
            equity.to_numpy(),
            label=f'{name} (Sharpe {sharpe:.2f})',
            **_style(name, result.candidate),
        )
    axis.axhline(1.0, color='black', linewidth=0.6)
    axis.set(
        title='Ciblage de volatilité sur le test final (portefeuille équipondéré, frais inclus)',
        xlabel='date',
        ylabel='valeur (base 1)',
    )
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def importance(result: TrainingResult) -> Figure | None:
    """Importance par permutation des features du candidat sur le test final."""
    values = result.importance
    if values is None or values.empty:
        return None
    top = values.head(TOP_FEATURES).iloc[::-1]
    figure, axis = plt.subplots(figsize=(9, 0.3 * len(top) + 1.5))
    axis.barh(top.index, top.to_numpy(), color=np.where(top.to_numpy() > 0, 'tab:blue', 'tab:red'))
    axis.set(
        title=f'Importance par permutation ({result.candidate}, test final, {TOP_FEATURES} premières)',
        xlabel='hausse de la MSE log-vol quand la feature est mélangée',
    )
    figure.tight_layout()
    return figure
