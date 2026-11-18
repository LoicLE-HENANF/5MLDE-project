"""Flow `ingest-candles` : historique (archives -> Parquet) puis bougies récentes (REST -> PostgreSQL).

Idempotent : un mois déjà en Parquet n'est pas retéléchargé, les bougies récentes reprennent après
la dernière stockée (upsert). Un lot rejeté n'arrête pas les autres ; le run finit en échec s'il y
en a eu au moins un, avec le détail dans tracing.ingestion_batches et l'artifact du run.

    uv run python -m flows.ingest                    # tous les symboles
    uv run python -m flows.ingest BTCUSDT --no-history
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from prefect import flow, task

from connectors.config import PostgresConfig, load_env
from connectors.postgres import PostgresConnector
from connectors.prefect import get_logger, publish_markdown, run_context, task_options
from flows.system import stopped
from ingestion.download import BinanceClient
from ingestion.pipeline import BatchProcessor, BatchResult, missing_months, recent_start
from ingestion.store import CandleTable, ParquetStore, data_dir
from marketdata.universe import SYMBOLS, check_symbol

FLOW_NAME = 'ingest-candles'
DEFAULT_RETENTION_DAYS = 45


class IngestionFailed(RuntimeError):
    """Au moins un lot a été rejeté."""


def retention(env: Mapping[str, str] | None = None) -> timedelta:
    raw = str((load_env() if env is None else env).get('CANDLES_RETENTION_DAYS', '')).strip()
    days = int(raw) if raw else DEFAULT_RETENTION_DAYS
    if days < 1:
        raise ValueError(f'CANDLES_RETENTION_DAYS={days} : au moins 1 jour attendu')
    return timedelta(days=days)


@task(name='history', **task_options())
def ingest_history(processor: BatchProcessor, symbol: str, now: datetime) -> list[BatchResult]:
    log = get_logger()
    results = []
    for month in missing_months(processor.parquet, symbol, now):
        result = processor.archive_month(symbol, month)
        if result.status != 'unavailable':
            log.info('%s', result)
            results.append(result)
    return results


@task(name='recent', **task_options())
def ingest_recent(processor: BatchProcessor, symbol: str, now: datetime, keep: timedelta) -> BatchResult:
    result = processor.recent(symbol, recent_start(processor.table, symbol, now, keep), now)
    pruned = processor.table.prune(symbol, now - keep)
    get_logger().info('%s (%d anciennes bougies purgées)', result, pruned)
    return result


def _report(results: Sequence[BatchResult]) -> str:
    lines = [
        '| Symbole | Source | Début | Fin | Statut | Bougies | Anomalies |',
        '|---|---|---|---|---|---|---|',
    ]
    for r in results:
        issues = '<br>'.join(str(i) for i in r.issues) or '-'
        lines.append(
            f'| {r.symbol} | {r.source} | {r.period_start:%Y-%m-%d %H:%M} | {r.period_end:%Y-%m-%d %H:%M} '
            f'| {r.status} | {r.rows} | {issues} |'
        )
    return '\n'.join(lines)


@flow(name=FLOW_NAME, log_prints=True)
def ingest_candles(symbols: list[str] | None = None, history: bool = True) -> dict[str, Any]:
    env = load_env()
    names = [check_symbol(s) for s in symbols or SYMBOLS]
    keep = retention(env)
    now = datetime.now(UTC)
    started = now
    flow_run_id = run_context().get('flow_run_id')
    results: list[BatchResult] = []
    with PostgresConnector(PostgresConfig.from_env(env)) as db, BinanceClient() as client:
        if (halted := stopped(db, FLOW_NAME, started, flow_run_id)) is not None:
            return halted
        processor = BatchProcessor(
            client,
            record=db.record_ingestion_batch,
            parquet=ParquetStore(data_dir(env)),
            table=CandleTable(db),
            flow_run_id=flow_run_id,
        )
        for symbol in names:
            if history:
                results.extend(ingest_history(processor, symbol, now))
            results.append(ingest_recent(processor, symbol, now, keep))
        rejected = [r for r in results if not r.accepted]
        summary = {
            'accepted': len(results) - len(rejected),
            'rejected': len(rejected),
            'rows': sum(r.rows for r in results if r.accepted),
        }
        db.record_pipeline_run(
            pipeline=FLOW_NAME,
            status='failure' if rejected else 'success',
            started_at=started,
            ended_at=datetime.now(UTC),
            prefect_flow_run_id=flow_run_id,
            details={'symbols': names, 'history': history, **summary},
        )
    publish_markdown(_report(results), key='ingestion', description='Lots ingérés par ce run')
    if rejected:
        raise IngestionFailed('lots rejetés :\n' + '\n'.join(str(r) for r in rejected))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Ingestion des bougies Binance (un run du flow).')
    parser.add_argument(
        'symbols', nargs='*', metavar='SYMBOL', help=f'parmi {", ".join(SYMBOLS)} (défaut : tous)'
    )
    parser.add_argument('--no-history', action='store_true', help='bougies récentes seulement')
    args = parser.parse_args(argv)
    try:
        print(ingest_candles(symbols=args.symbols or None, history=not args.no_history))
    except IngestionFailed as exc:
        print(exc)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
