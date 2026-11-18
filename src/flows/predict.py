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
from serving import Forecast, predict_latest, predict_latest_returns
from training.returns import HORIZON_HOURS as RETURN_HORIZON_HOURS
from training.returns import MODEL_NAME as RETURN_MODEL_NAME
from training.volatility import HORIZON_HOURS, MODEL_NAME

FLOW_NAME = 'predict-volatility'
RETURN_FLOW_NAME = 'predict-returns'
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
    connector: MlflowConnector,
    alias: str,
    candles: pd.DataFrame,
    now: datetime,
    model_name: str = MODEL_NAME,
    horizon_hours: int = HORIZON_HOURS,
) -> tuple[Any, Forecast]:
    version = connector.get_version(model_name, alias)
    model = connector.load_model(model_name, alias, loader=mlflow.pyfunc.load_model)
    if model_name == MODEL_NAME:
        infer = predict_latest
    elif model_name == RETURN_MODEL_NAME:
        infer = predict_latest_returns
    else:
        raise ValueError(f'modèle non pris en charge par le flow predict : {model_name!r}')
    result = infer(model.predict, candles, as_of=now, with_positions=alias == CHAMPION)
    log = get_logger()
    log.info(
        '%s %s v%s (run MLflow %s) : %d prédictions sur la bougie %s, horizon %d h',
        model_name,
        alias,
        version.version,
        version.run_id,
        len(result.frame),
        f'{result.open_time:%Y-%m-%d %H:%M}',
        horizon_hours,
    )
    for row in result.frame.itertuples(index=False):
        position = '-' if pd.isna(row.position) else f'{row.position:.3f}'
        baseline = '-' if pd.isna(row.baseline) else f'{row.baseline:.4f}'
        log.info(
            '  %s : prévision %.4f (référence %s), position %s',
            row.symbol,
            row.prediction,
            baseline,
            position,
        )
    return version, result


def record(
    db: PostgresConnector,
    version: Any,
    alias: str,
    result: Forecast,
    flow_run_id: str | None,
    model_name: str = MODEL_NAME,
    horizon_hours: int = HORIZON_HOURS,
) -> int:
    """Trace chaque prédiction ; renvoie le nombre de lignes nouvelles."""
    inserted = 0
    for row in result.frame.itertuples(index=False):
        identifier = db.record_prediction(
            model_name=model_name,
            version=int(version.version),
            alias=alias,
            symbol=row.symbol,
            open_time=row.open_time.to_pydatetime(),
            horizon_hours=horizon_hours,
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


def _report(results: Mapping[str, tuple[Any, Forecast]], horizon_hours: int) -> str:
    lines = [
        f'| Alias | Version | Symbole | Bougie | Prévision {horizon_hours} h | Référence | Position |',
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


def _predict_model(
    model_name: str,
    horizon_hours: int,
    flow_name: str,
    as_of: datetime | None,
) -> dict[str, Any]:
    env = load_env()
    now = datetime.now(UTC) if as_of is None else as_of
    started = datetime.now(UTC)
    flow_run_id = run_context().get('flow_run_id')
    connector = MlflowConnector(MlflowConfig.from_env(env))
    with PostgresConnector(PostgresConfig.from_env(env)) as db:
        if (halted := stopped(db, flow_name, started, flow_run_id)) is not None:
            return halted
        if connector.get_version(model_name, CHAMPION) is None:
            db.record_pipeline_run(
                pipeline=flow_name,
                status='cancelled',
                started_at=started,
                ended_at=datetime.now(UTC),
                prefect_flow_run_id=flow_run_id,
                details={'raison': f'aucun {CHAMPION} pour {model_name}'},
            )
            raise NoChampion(f'aucune version {CHAMPION!r} de {model_name} : entraîner puis promouvoir')
        candles = recent_candles(db, now)
        get_logger().info(
            'référence %s UTC ; %d bougies chargées (%d symboles, dernière clôturée %s)',
            f'{now:%Y-%m-%d %H:%M}',
            len(candles),
            candles['symbol'].nunique(),
            f'{candles["open_time"].max():%Y-%m-%d %H:%M}',
        )
        aliases = [CHAMPION] + ([CHALLENGER] if connector.get_version(model_name, CHALLENGER) else [])
        results = {
            alias: forecast(connector, alias, candles, now, model_name, horizon_hours) for alias in aliases
        }
        inserted = {
            alias: record(db, version, alias, result, flow_run_id, model_name, horizon_hours)
            for alias, (version, result) in results.items()
        }
        champion_version, champion = results[CHAMPION]
        summary = {
            'modele': model_name,
            'bougie': champion.open_time.isoformat(),
            'versions': {alias: int(version.version) for alias, (version, _) in results.items()},
            'nouvelles': inserted,
        }
        db.record_pipeline_run(
            pipeline=flow_name,
            status='success',
            started_at=started,
            ended_at=datetime.now(UTC),
            prefect_flow_run_id=flow_run_id,
            mlflow_run_id=champion_version.run_id,
            details=summary,
        )
    get_logger().info('bilan : %s', summary)
    publish_markdown(
        _report(results, horizon_hours),
        key=f'predictions-{model_name}',
        description=f'Prévisions {model_name} de ce run',
    )
    return summary


@flow(name=FLOW_NAME, log_prints=True)
def predict_volatility(as_of: datetime | None = None) -> dict[str, Any]:
    return _predict_model(MODEL_NAME, HORIZON_HOURS, FLOW_NAME, as_of)


@flow(name=RETURN_FLOW_NAME, log_prints=True)
def predict_returns(as_of: datetime | None = None) -> dict[str, Any]:
    return _predict_model(RETURN_MODEL_NAME, RETURN_HORIZON_HOURS, RETURN_FLOW_NAME, as_of)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Prévision en production (un run du flow).')
    parser.add_argument(
        '--model',
        choices=(MODEL_NAME, RETURN_MODEL_NAME),
        default=MODEL_NAME,
        help='modèle à servir',
    )
    parser.add_argument(
        '--as-of',
        type=datetime.fromisoformat,
        default=None,
        help="instant d'observation, avec fuseau (défaut : maintenant)",
    )
    args = parser.parse_args(argv)
    try:
        predict = predict_returns if args.model == RETURN_MODEL_NAME else predict_volatility
        print(predict(as_of=args.as_of))
    except NoChampion as exc:
        print(exc)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
