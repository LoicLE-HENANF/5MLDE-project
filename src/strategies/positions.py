"""Positions simulées construites à partir des prédictions des modèles.

Les prédictions arrivent au format long (`symbol`, `open_time`, colonne de prédiction) : une ligne
par bougie, calculée une fois la bougie close. Les positions sont renvoyées au format large :
une ligne par heure (`open_time`), une colonne par symbole, un poids par case (fraction du
capital, positif = acheteur, négatif = vendeur, 0 = cash). Une prédiction absente donne un poids nul.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_FEE = 0.001  # 0,1 % du montant échangé, frais taker spot Binance


def to_wide(predictions: pd.DataFrame, column: str = 'prediction') -> pd.DataFrame:
    """Prédictions longues -> tableau heure x symbole (NaN si le symbole n'a pas de prédiction)."""
    missing = [name for name in ('symbol', 'open_time', column) if name not in predictions.columns]
    if missing:
        raise ValueError(f'colonnes absentes : {missing}')
    if predictions.empty:
        raise ValueError('aucune prédiction')
    if not isinstance(predictions['open_time'].dtype, pd.DatetimeTZDtype):
        raise ValueError('open_time doit contenir des horodatages avec fuseau')
    if predictions.duplicated(['symbol', 'open_time']).any():
        raise ValueError('doublons sur (symbol, open_time)')
    values = pd.to_numeric(predictions[column], errors='raise').astype('float64')
    if np.isinf(values).any():
        raise ValueError(f'{column} contient des valeurs infinies')
    wide = predictions.assign(**{column: values}).pivot(index='open_time', columns='symbol', values=column)
    wide.columns.name = 'symbol'
    return wide.sort_index().sort_index(axis=1)


def long_short(predictions: pd.DataFrame, column: str = 'prediction') -> pd.DataFrame:
    """Long/short transversal : poids proportionnels au rang centré des prédictions de l'heure.

    Les symboles au rendement prédit le plus haut sont achetés, les plus bas vendus ; à chaque
    heure la somme des poids est nulle (neutre au marché) et la somme des valeurs absolues vaut 1.
    Une heure avec moins de deux prédictions, ou des prédictions toutes égales, reste en cash.
    """
    wide = to_wide(predictions, column)
    ranks = wide.rank(axis=1)
    centered = ranks.sub(ranks.mean(axis=1), axis=0)
    gross = centered.abs().sum(axis=1)
    weights = centered.div(gross.where(gross > 0), axis=0)
    return weights.fillna(0.0)


def long_cash(
    predictions: pd.DataFrame, column: str = 'prediction', threshold: float = DEFAULT_FEE
) -> pd.DataFrame:
    """Long/cash par symbole : part égale du capital si le rendement prédit dépasse `threshold`.

    Le seuil par défaut vaut les frais d'un ordre : en dessous, entrer en position coûte plus
    que le gain attendu. Chaque symbole reçoit au plus 1/n du capital (n = symboles prédits).
    """
    if not np.isfinite(threshold):
        raise ValueError('threshold doit être fini')
    wide = to_wide(predictions, column)
    share = 1.0 / wide.shape[1]
    return (wide > threshold).astype('float64') * share


def equal_weight(index: pd.Index, symbols: list[str] | pd.Index) -> pd.DataFrame:
    """Achat-conservation équipondéré : 1/n du capital sur chaque symbole à chaque heure."""
    symbols = sorted(symbols)
    if not symbols:
        raise ValueError('aucun symbole')
    weights = pd.DataFrame(1.0 / len(symbols), index=index, columns=pd.Index(symbols, name='symbol'))
    return weights


def scale_by_volatility(
    positions: pd.DataFrame,
    volatility: pd.DataFrame,
    *,
    target: float,
    max_leverage: float = 1.0,
) -> pd.DataFrame:
    """Ciblage de volatilité : chaque poids est multiplié par target / volatilité prédite.

    Le multiplicateur est borné à [0, max_leverage] (1 = jamais d'effet de levier). Sans
    prévision de volatilité pour une case, la position est coupée (cash). `volatility` est au
    format large (voir `to_wide`) et dans la même unité que `target`.
    """
    if not (np.isfinite(target) and target > 0):
        raise ValueError('target doit être strictement positif')
    if not (np.isfinite(max_leverage) and max_leverage > 0):
        raise ValueError('max_leverage doit être strictement positif')
    aligned = volatility.reindex(index=positions.index, columns=positions.columns)
    if (aligned <= 0).any().any():
        raise ValueError('les volatilités prédites doivent être strictement positives')
    multiplier = (target / aligned).clip(upper=max_leverage).fillna(0.0)
    return positions * multiplier
