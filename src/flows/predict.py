"""Flow `predict-volatility` : prévision 24 h de la dernière bougie clôturée de chaque symbole.

Le champion du registre MLflow prédit et fixe la position de la stratégie de ciblage ;
le challenger, s'il existe, prédit en parallèle sans position. Chaque prédiction est
tracée dans `tracing.predictions` avec sa version, son run MLflow et la référence
naïve (persistance). Rejouer la même heure ne crée pas de doublon.

    uv run python -m flows.predict
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import mlflow
import pandas as pd
from prefect import flow, task

from connectors.config import MlflowConfig, PostgresConfig, load_env
from connectors.mlflow import MlflowConnector
from connectors.postgres import PostgresConnector
from connectors.prefect import get_logger, publish_markdown, run_context, task_options
from flows.system import stopped
from ingestion.store import CandleTable
from registry import CHALLENGER, CHAMPION
from serving import Forecast, predict_latest
from training.volatility import HORIZON_HOURS, MODEL_NAME

FLOW_NAME = 'predict-volatility'
# Historique lu dans PostgreSQL : contexte des features (168 h) et volatilité cible (30 jours).
LOOKBACK = timedelta(days=32)


class NoChampion(RuntimeError):
    """Aucune version n'est en service : rien à prédire."""


def recent_candles(db: PostgresConnector, now: datetime) -> pd.DataFrame:
    candles = CandleTable(db).read(start=now - LOOKBACK)
    if candles.empty:
        raise ValueError('aucune bougie récente dans market.candles : lancer flows.ingest')
    return candles


@task(name='forecast', **task_options())
def forecast(
    connector: MlflowConnector, alias: str, candles: pd.DataFrame, now: datetime
) -> tuple[Any, Forecast]:
    version = connector.get_version(MODEL_NAME, alias)
    model = connector.load_model(MODEL_NAME, alias, loader=mlflow.pyfunc.load_model)
    result = predict_latest(model.predict, candles, as_of=now, with_positions=alias == CHAMPION)
    log = get_logger()
    log.info(
        '%s v%s (run MLflow %s) : %d prédictions sur la bougie %s, horizon %d h',
        alias,
        version.version,
        version.run_id,
        len(result.frame),
        f'{result.open_time:%Y-%m-%d %H:%M}',
        HORIZON_HOURS,
    )
    for row in result.frame.itertuples(index=False):
        position = '-' if pd.isna(row.position) else f'{row.position:.3f}'
        baseline = '-' if pd.isna(row.baseline) else f'{row.baseline:.4f}'
        log.info(
            '  %s : volatilité prévue %.4f (persistance %s), position %s',
            row.symbol,
            row.prediction,
            baseline,
            position,
        )
    return version, result


def record(db: PostgresConnector, version: Any, alias: str, result: Forecast, flow_run_id: str | None) -> int:
    """Trace chaque prédiction ; renvoie le nombre de lignes nouvelles."""
    inserted = 0
    for row in result.frame.itertuples(index=False):
        identifier = db.record_prediction(
            model_name=MODEL_NAME,
            version=int(version.version),
            alias=alias,
            symbol=row.symbol,
            open_time=row.open_time.to_pydatetime(),
            horizon_hours=HORIZON_HOURS,
            prediction=float(row.prediction),
            baseline=None if pd.isna(row.baseline) else float(row.baseline),
            position=None if pd.isna(row.position) else float(row.position),
            mode='live',
            mlflow_run_id=version.run_id,
            prefect_flow_run_id=flow_run_id,
        )
        inserted += identifier is not None
    duplicates = len(result.frame) - inserted
    get_logger().info(
        '%s : %d prédictions tracées dans tracing.predictions%s',
        alias,
        inserted,
        f', {duplicates} déjà présentes (rejeu ignoré)' if duplicates else '',
    )
    return inserted


def _report(results: Mapping[str, tuple[Any, Forecast]]) -> str:
    lines = [
        '| Alias | Version | Symbole | Bougie | Prévision 24 h | Persistance | Position |',
        '|---|---|---|---|---|---|---|',
    ]
    for alias, (version, result) in results.items():
        for row in result.frame.itertuples(index=False):
            position = '-' if pd.isna(row.position) else f'{row.position:.3f}'
            lines.append(
                f'| {alias} | {version.version} | {row.symbol} | {row.open_time:%Y-%m-%d %H:%M} '
                f'| {row.prediction:.4f} | {row.baseline:.4f} | {position} |'
            )
    return '\n'.join(lines)


@flow(name=FLOW_NAME, log_prints=True)
def predict_volatility(as_of: datetime | None = None) -> dict[str, Any]:
    env = load_env()
    now = datetime.now(UTC) if as_of is None else as_of
    started = datetime.now(UTC)
    flow_run_id = run_context().get('flow_run_id')
    connector = MlflowConnector(MlflowConfig.from_env(env))
    with PostgresConnector(PostgresConfig.from_env(env)) as db:
        if (halted := stopped(db, FLOW_NAME, started, flow_run_id)) is not None:
            return halted
        if connector.get_version(MODEL_NAME, CHAMPION) is None:
            db.record_pipeline_run(
                pipeline=FLOW_NAME,
                status='cancelled',
                started_at=started,
                ended_at=datetime.now(UTC),
                prefect_flow_run_id=flow_run_id,
                details={'raison': f'aucun {CHAMPION} pour {MODEL_NAME}'},
            )
            raise NoChampion(f'aucune version {CHAMPION!r} de {MODEL_NAME} : entraîner puis promouvoir')
        candles = recent_candles(db, now)
        get_logger().info(
            'référence %s UTC ; %d bougies chargées (%d symboles, dernière clôturée %s)',
            f'{now:%Y-%m-%d %H:%M}',
            len(candles),
            candles['symbol'].nunique(),
            f'{candles["open_time"].max():%Y-%m-%d %H:%M}',
        )
        aliases = [CHAMPION] + ([CHALLENGER] if connector.get_version(MODEL_NAME, CHALLENGER) else [])
        results = {alias: forecast(connector, alias, candles, now) for alias in aliases}
        inserted = {
            alias: record(db, version, alias, result, flow_run_id)
            for alias, (version, result) in results.items()
        }
        champion_version, champion = results[CHAMPION]
        summary = {
            'bougie': champion.open_time.isoformat(),
            'versions': {alias: int(version.version) for alias, (version, _) in results.items()},
            'nouvelles': inserted,
        }
        db.record_pipeline_run(
            pipeline=FLOW_NAME,
            status='success',
            started_at=started,
            ended_at=datetime.now(UTC),
            prefect_flow_run_id=flow_run_id,
            mlflow_run_id=champion_version.run_id,
            details=summary,
        )
    get_logger().info('bilan : %s', summary)
    publish_markdown(_report(results), key='predictions', description='Prévisions de ce run')
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Prévision de volatilité (un run du flow).')
    parser.add_argument(
        '--as-of',
        type=datetime.fromisoformat,
        default=None,
        help="instant d'observation, avec fuseau (défaut : maintenant)",
    )
    args = parser.parse_args(argv)
    try:
        print(predict_volatility(as_of=args.as_of))
    except NoChampion as exc:
        print(exc)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
