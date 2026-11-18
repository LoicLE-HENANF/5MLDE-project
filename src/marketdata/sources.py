"""Les deux sources de bougies et leur conversion vers le format normalisé.

Archives `data.binance.vision` : un zip mensuel par symbole contenant un CSV sans en-tête
(certains fichiers en ont une, elle est ignorée), accompagné d'un `.CHECKSUM` (SHA-256).
API REST `/api/v3/klines` : liste de listes JSON, prix et volumes en chaînes, au plus
1000 bougies par appel.

Les deux renvoient les mêmes 12 champs. Les horodatages sont en millisecondes, sauf dans
les archives depuis 2025-01 où ils sont en microsecondes : l'unité est déduite de la
magnitude de chaque valeur, puis tout est converti en UTC à la microseconde.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence
from datetime import date
from typing import Any

import pandas as pd

from marketdata.universe import INTERVAL, INTERVAL_DELTA, check_symbol

ARCHIVE_BASE_URL = 'https://data.binance.vision/data/spot/monthly/klines'
REST_KLINES_URL = 'https://data-api.binance.vision/api/v3/klines'
REST_MAX_LIMIT = 1000

# Ordre des champs, identique dans les archives et l'API REST.
RAW_FIELDS = (
    'open_time',
    'open',
    'high',
    'low',
    'close',
    'volume',
    'close_time',
    'quote_volume',
    'trades',
    'taker_buy_base_volume',
    'taker_buy_quote_volume',
    'ignore',
)
PRICE_FIELDS = ('open', 'high', 'low', 'close')
VOLUME_FIELDS = ('volume', 'quote_volume', 'taker_buy_base_volume', 'taker_buy_quote_volume')
TIME_FIELDS = ('open_time', 'close_time')

# Au-delà de ce seuil un horodatage epoch est en microsecondes (1e14 ms = an 5138,
# 1e14 µs = 1973) : aucune bougie Binance n'est ambiguë.
_MICROSECONDS_FROM = 10**14
_ONE_US = pd.Timedelta(microseconds=1)


class SourceFormatError(ValueError):
    """Les données reçues ne respectent pas le format attendu de la source."""


def archive_url(symbol: str, month: date) -> str:
    """URL du zip mensuel ; le `.CHECKSUM` est à la même adresse suffixée."""
    symbol = check_symbol(symbol)
    name = f'{symbol}-{INTERVAL}-{month:%Y-%m}.zip'
    return f'{ARCHIVE_BASE_URL}/{symbol}/{INTERVAL}/{name}'


def checksum_url(symbol: str, month: date) -> str:
    return archive_url(symbol, month) + '.CHECKSUM'


def parse_checksum(text: str) -> str:
    """Extrait le SHA-256 d'un fichier `.CHECKSUM` (« <hex>  <nom du fichier> »)."""
    parts = text.split()
    digest = parts[0].lower() if parts else ''
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise SourceFormatError(f'.CHECKSUM illisible : {text[:80]!r}')
    return digest


def rest_params(symbol: str, start_ms: int, end_ms: int | None = None) -> dict[str, Any]:
    """Paramètres d'un appel `/api/v3/klines` (bornes en millisecondes, incluses)."""
    params: dict[str, Any] = {
        'symbol': check_symbol(symbol),
        'interval': INTERVAL,
        'startTime': start_ms,
        'limit': REST_MAX_LIMIT,
    }
    if end_ms is not None:
        params['endTime'] = end_ms
    return params


def read_archive_csv(content: bytes | str) -> list[list[str]]:
    """Lignes brutes du CSV extrait d'une archive (en-tête éventuelle retirée)."""
    text = content.decode('utf-8') if isinstance(content, bytes) else content
    rows = [row for row in csv.reader(io.StringIO(text)) if row]
    if rows and not rows[0][0].strip().isdigit():
        rows = rows[1:]
    return rows


def normalize(rows: Iterable[Sequence[Any]], symbol: str) -> pd.DataFrame:
    """Lignes brutes (archive ou REST) → bougies normalisées, triées par `open_time`.

    Le champ `ignore` est retiré, le symbole ajouté en première colonne. Lève
    SourceFormatError si une ligne n'a pas 12 champs, contient une valeur non numérique
    ou si une bougie ne dure pas exactement une heure (bougies interrompues comprises).
    """
    frame, interrupted = normalize_dropping_interrupted(rows, symbol)
    if interrupted:
        raise SourceFormatError(
            f'{check_symbol(symbol)} : {len(interrupted)} bougie(s) dont close_time ne correspond pas à '
            f'un intervalle de {INTERVAL} (première : {interrupted[0]})'
        )
    return frame


def normalize_dropping_interrupted(
    rows: Iterable[Sequence[Any]], symbol: str
) -> tuple[pd.DataFrame, tuple[pd.Timestamp, ...]]:
    """Comme `normalize`, mais retire les bougies interrompues au lieu de lever.

    Une bougie interrompue se termine avant la fin de son heure (`open_time <= close_time
    < open_time + 1 h - 1 ms`) : Binance l'émet quand le trading s'arrête en cours d'heure,
    par exemple le 2023-03-24 à 12:00 UTC (clôture à 12:39, volume nul). Elle ne décrit
    pas une heure complète, elle est donc écartée ; ses `open_time` sont renvoyés pour être
    tracés. Toute autre durée (clôture avant l'ouverture ou après la fin de l'heure)
    reste une erreur de format.
    """
    symbol = check_symbol(symbol)
    rows = list(rows)
    bad = [i for i, row in enumerate(rows) if len(row) != len(RAW_FIELDS)]
    if bad:
        raise SourceFormatError(
            f'{symbol} : {len(bad)} ligne(s) sans {len(RAW_FIELDS)} champs (première : {bad[0]})'
        )
    frame = pd.DataFrame(rows, columns=list(RAW_FIELDS)).drop(columns='ignore')
    try:
        for name in TIME_FIELDS:
            frame[name] = to_utc(frame[name])
        for name in PRICE_FIELDS + VOLUME_FIELDS:
            frame[name] = pd.to_numeric(frame[name], errors='raise').astype('float64')
        frame['trades'] = pd.to_numeric(frame['trades'], errors='raise').astype('int64')
    except (TypeError, ValueError) as exc:
        raise SourceFormatError(f'{symbol} : valeur non numérique ({exc})') from exc
    # Fin de bougie : open + 1 h - 1 ms (source en ms) ou - 1 µs (archives depuis 2025).
    # Toute autre durée signale un mauvais intervalle ; la valeur est ensuite ramenée à
    # open + 1 h - 1 µs pour qu'une même bougie soit identique quelle que soit sa source.
    end = frame['open_time'] + INTERVAL_DELTA
    complete_from = end - pd.Timedelta(milliseconds=1)
    complete = frame['close_time'].between(complete_from, end - _ONE_US)
    interrupted = (frame['close_time'] >= frame['open_time']) & (frame['close_time'] < complete_from)
    wrong = ~(complete | interrupted)
    if wrong.any():
        raise SourceFormatError(
            f'{symbol} : {int(wrong.sum())} bougie(s) dont close_time ne correspond pas à '
            f'un intervalle de {INTERVAL} (première : {frame.loc[wrong, "open_time"].iloc[0]})'
        )
    dropped = tuple(sorted(frame.loc[interrupted, 'open_time']))
    frame = frame.loc[complete].copy()
    frame['close_time'] = frame['open_time'] + INTERVAL_DELTA - _ONE_US
    frame.insert(0, 'symbol', pd.Series([symbol] * len(frame), index=frame.index, dtype='str'))
    return frame.sort_values('open_time', ignore_index=True), dropped


def to_utc(values: pd.Series) -> pd.Series:
    """Horodatages epoch en ms ou µs (mélangés possibles) → datetime UTC à la µs."""
    epoch = pd.to_numeric(values, errors='raise').astype('int64')
    micros = epoch.where(epoch >= _MICROSECONDS_FROM, epoch * 1000)
    return pd.to_datetime(micros, unit='us', utc=True).astype('datetime64[us, UTC]')
