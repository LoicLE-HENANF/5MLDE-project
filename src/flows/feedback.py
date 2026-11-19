"""Flow `feedback-volatility` : retours sur les prédictions à horizon clôturé et santé des modèles (niveau 3).

1. Chaque prédiction dont l'horizon est clôturé reçoit sa volatilité réalisée et ses
   pertes QLIKE (modèle et persistance) dans `tracing.feedback`.
2. Pour le dernier jour complet, une ligne de santé par version servie est écrite
   dans `tracing.model_health` : QLIKE glissante sur 7 jours face à la persistance,
   PnL simulé des positions du champion face à l'équipondéré.
3. Si le champion est dégradé `ROLLBACK_AFTER_DAYS` jours de suite, l'ancien
   champion (`precedent`) est remis en service et l'événement tracé.

    uv run python -m flows.feedback
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
from prefect import flow, task

from connectors.config import MlflowConfig, PostgresConfig, load_env
from connectors.mlflow import MlflowConnector
from connectors.postgres import PostgresConnector
from connectors.prefect import get_logger, publish_markdown, run_context, task_options
from flows.system import stopped
from ingestion.store import CandleTable
from marketdata.universe import INTERVAL_DELTA
from registry import CHAMPION, RegistryError, rollback
from serving import (
    DEFAULT_WINDOW_HOURS,
    health_rows,
    pending_feedback,
    realized_feedback,
    realized_return_feedback,
    return_health_rows,
)
from training.returns import MODEL_NAME as RETURN_MODEL_NAME
from training.volatility import MODEL_NAME

FLOW_NAME = 'feedback-volatility'
RETURN_FLOW_NAME = 'feedback-returns'
ROLLBACK_AFTER_DAYS = 3
# Une journée D est complète quand les prédictions de sa dernière heure ont leur retour.
HEALTH_DELAY = timedelta(hours=48)

PENDING_SQL = """
SELECT p.id AS prediction_id, p.symbol, p.open_time, p.horizon_hours, p.prediction, p.baseline
FROM tracing.predictions p
LEFT JOIN tracing.feedback f ON f.prediction_id = p.id
WHERE f.id IS NULL AND p.model_name = %s AND p.open_time >= %s
ORDER BY p.open_time
"""

SCORED_SQL = """
SELECT p.model_name, p.version, p.alias, p.mode, p.symbol, p.open_time, p.position,
       p.prediction, p.baseline, f.realized, f.loss, f.baseline_loss
FROM tracing.predictions p
JOIN tracing.feedback f ON f.prediction_id = p.id
WHERE p.model_name = %s AND p.open_time >= %s AND p.open_time < %s
"""

HEALTH_SQL = """
SELECT day, degraded FROM tracing.model_health
WHERE model_name = %s AND version = %s AND mode = 'live' AND metric = %s
ORDER BY day DESC
LIMIT %s
"""


@task(name='feedback', **task_options())
def score_pending(
    db: PostgresConnector,
    candles: pd.DataFrame,
    now: datetime,
    since: datetime,
    flow_run_id: str | None,
    model_name: str = MODEL_NAME,
) -> int:
    log = get_logger()
    waiting = db.select(PENDING_SQL, [model_name, since])
    pending = pending_feedback(waiting, pd.Timestamp(now))
    log.info(
        'prédictions sans retour depuis le %s : %d, dont %d à horizon clôturé',
        f'{since:%Y-%m-%d %H:%M}',
        len(waiting),
        len(pending),
    )
    if len(waiting) > len(pending):
        _log_next_due(log, waiting, pending)
    feedback = realized_return_feedback if model_name == RETURN_MODEL_NAME else realized_feedback
    scored = feedback(pending, candles, as_of=pd.Timestamp(now))
    for row in scored.itertuples(index=False):
        db.record_feedback(
            prediction_id=int(row.prediction_id),
            realized=float(row.realized),
            loss=float(row.loss),
            baseline_loss=None if pd.isna(row.baseline_loss) else float(row.baseline_loss),
            prefect_flow_run_id=flow_run_id,
        )
    skipped = len(pending) - len(scored)
    if skipped:
        log.warning(
            '%d prédictions sans valeur réalisée (bougie manquante), retentées plus tard',
            skipped,
        )
    if len(scored):
        model = scored['loss'].astype('float64').mean()
        baseline = scored['baseline_loss'].astype('float64').mean()
        # perte par prédiction ; la santé du rendement se juge sur l'IC (feedback_returns)
        label = 'erreur quadratique' if model_name == RETURN_MODEL_NAME else 'QLIKE'
        log.info(
            '%d retours écrits : %s moyenne %.4f (référence %.4f) ; prédictions du %s au %s',
            len(scored),
            label,
            model,
            baseline,
            f'{pending["open_time"].min():%Y-%m-%d %H:%M}',
            f'{pending["open_time"].max():%Y-%m-%d %H:%M}',
        )
    else:
        log.info('aucun retour écrit')
    return len(scored)


def _log_next_due(log: Any, waiting: Sequence[Mapping[str, Any]], pending: pd.DataFrame) -> None:
    """Indique quand la prochaine prédiction en attente aura son retour."""
    done = set(pending['prediction_id']) if not pending.empty else set()
    due = [
        pd.Timestamp(row['open_time']) + INTERVAL_DELTA + pd.Timedelta(hours=int(row['horizon_hours']))
        for row in waiting
        if row['prediction_id'] not in done
    ]
    log.info(
        '%d prédictions encore en attente de leur horizon ; prochain retour possible le %s UTC',
        len(due),
        f'{min(due):%Y-%m-%d %H:%M}',
    )


@task(name='health', **task_options())
def record_health(
    db: PostgresConnector,
    candles: pd.DataFrame,
    day: date,
    flow_run_id: str | None,
    model_name: str = MODEL_NAME,
) -> list[dict[str, Any]]:
    end = datetime.combine(day, datetime.min.time(), tzinfo=UTC) + timedelta(days=1)
    start = end - timedelta(hours=DEFAULT_WINDOW_HOURS)
    log = get_logger()
    scored = pd.DataFrame(db.select(SCORED_SQL, [model_name, start, end]))
    log.info(
        'santé du %s : fenêtre %s -> %s (%d h), %d prédictions avec retour',
        day.isoformat(),
        f'{start:%Y-%m-%d %H:%M}',
        f'{end:%Y-%m-%d %H:%M}',
        DEFAULT_WINDOW_HOURS,
        len(scored),
    )
    if model_name == RETURN_MODEL_NAME:
        rows = return_health_rows(scored, candles, day=day)
    else:
        rows = health_rows(scored, candles, day=day)
    if not rows:
        log.info('aucune version à évaluer : pas encore de retours dans la fenêtre')
    for row in rows:
        db.record_model_health(**row, prefect_flow_run_id=flow_run_id)
        (log.warning if row['degraded'] else log.info)(
            'v%s %s : %d échantillons, %s %s vs référence %s, PnL %s vs équipondéré %s -> %s',
            row['version'],
            row['mode'],
            row['samples'],
            row['metric'],
            _fmt(row['value'], '.4f'),
            _fmt(row['baseline_value'], '.4f'),
            _fmt(row['pnl'], '+.2%'),
            _fmt(row['baseline_pnl'], '+.2%'),
            'DÉGRADÉE' if row['degraded'] else 'saine',
        )
    return rows


def _fmt(value: float | None, pattern: str) -> str:
    return '-' if value is None else format(value, pattern)


def should_roll_back(history: Sequence[dict[str, Any]], days: int = ROLLBACK_AFTER_DAYS) -> bool:
    """Vrai si les `days` derniers jours (consécutifs, du plus récent au plus ancien) sont dégradés."""
    if len(history) < days:
        return False
    recent = list(history)[:days]
    consecutive = all((recent[i]['day'] - recent[i + 1]['day']).days == 1 for i in range(len(recent) - 1))
    return consecutive and all(row['degraded'] for row in recent)


@task(name='rollback', **task_options())
def check_rollback(
    db: PostgresConnector,
    connector: MlflowConnector,
    model_name: str = MODEL_NAME,
    metric: str = 'qlike',
) -> int | None:
    """Retour arrière automatique du champion dégradé ; renvoie la version restaurée."""
    log = get_logger()
    champion = connector.get_version(model_name, CHAMPION)
    if champion is None:
        log.info('aucun champion %s : pas de retour arrière à évaluer', model_name)
        return None
    history = db.select(
        HEALTH_SQL,
        [model_name, int(champion.version), metric, ROLLBACK_AFTER_DAYS],
    )
    if not should_roll_back(history):
        degraded = sum(1 for row in history if row['degraded'])
        log.info(
            'champion v%s : %d jour(s) de santé connus, %d dégradé(s) ; '
            'retour arrière après %d jours dégradés consécutifs',
            champion.version,
            len(history),
            degraded,
            ROLLBACK_AFTER_DAYS,
        )
        return None
    try:
        restored = rollback(connector, model_name, record=db.record_model_event)
    except RegistryError as exc:
        log.warning(
            'champion v%s dégradé %d jours, retour arrière impossible : %s',
            champion.version,
            ROLLBACK_AFTER_DAYS,
            exc,
        )
        return None
    log.warning(
        'champion v%s dégradé %d jours de suite : retour à la v%s',
        champion.version,
        ROLLBACK_AFTER_DAYS,
        restored.version,
    )
    return int(restored.version)


def _report(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        '| Version | Mode | Jour | N | Métrique | Référence | PnL | PnL équipondéré | Dégradé |',
        '|---|---|---|---|---|---|---|---|---|',
    ]

    def fmt(value: float | None, pattern: str) -> str:
        return '-' if value is None else format(value, pattern)

    for r in rows:
        lines.append(
            f'| {r["version"]} | {r["mode"]} | {r["day"]} | {r["samples"]} | {r["metric"]} '
            f'{fmt(r["value"], ".4f")} '
            f'| {fmt(r["baseline_value"], ".4f")} | {fmt(r["pnl"], ".2%")} | {fmt(r["baseline_pnl"], ".2%")} '
            f'| {"oui" if r["degraded"] else "non"} |'
        )
    return '\n'.join(lines)


def _feedback_model(
    model_name: str,
    flow_name: str,
    metric: str,
    as_of: datetime | None,
) -> dict[str, Any]:
    env = load_env()
    now = datetime.now(UTC) if as_of is None else as_of
    started = datetime.now(UTC)
    flow_run_id = run_context().get('flow_run_id')
    day = (now - HEALTH_DELAY).date()
    since = now - timedelta(days=30)
    connector = MlflowConnector(MlflowConfig.from_env(env))
    with PostgresConnector(PostgresConfig.from_env(env)) as db:
        if (halted := stopped(db, flow_name, started, flow_run_id)) is not None:
            return halted
        window_start = datetime.combine(day, datetime.min.time(), tzinfo=UTC) - timedelta(
            hours=DEFAULT_WINDOW_HOURS
        )
        candles = CandleTable(db).read(start=min(since, window_start))
        log = get_logger()
        if candles.empty:
            log.warning(
                'aucune bougie en base depuis le %s : lancer flows.ingest',
                f'{min(since, window_start):%Y-%m-%d}',
            )
        else:
            log.info(
                'référence %s UTC ; %d bougies chargées (%d symboles, dernière clôturée %s)',
                f'{now:%Y-%m-%d %H:%M}',
                len(candles),
                candles['symbol'].nunique(),
                f'{candles["open_time"].max():%Y-%m-%d %H:%M}',
            )
        written = score_pending(db, candles, now, since, flow_run_id, model_name)
        health = record_health(db, candles, day, flow_run_id, model_name)
        restored = check_rollback(db, connector, model_name, metric)
        summary = {
            'modele': model_name,
            'retours': written,
            'jour_sante': day.isoformat(),
            'versions_suivies': len(health),
            'degradees': sum(1 for r in health if r['degraded']),
            'retour_arriere': restored,
        }
        log.info('bilan : %s', summary)
        db.record_pipeline_run(
            pipeline=flow_name,
            status='success',
            started_at=started,
            ended_at=datetime.now(UTC),
            prefect_flow_run_id=flow_run_id,
            details=summary,
        )
    if health:
        publish_markdown(
            _report(health),
            key=f'sante-{model_name}',
            description=f'Santé (niveau 3) du modèle {model_name}',
        )
    return summary


@flow(name=FLOW_NAME, log_prints=True)
def feedback_volatility(as_of: datetime | None = None) -> dict[str, Any]:
    return _feedback_model(MODEL_NAME, FLOW_NAME, 'qlike', as_of)


@flow(name=RETURN_FLOW_NAME, log_prints=True)
def feedback_returns(as_of: datetime | None = None) -> dict[str, Any]:
    return _feedback_model(RETURN_MODEL_NAME, RETURN_FLOW_NAME, 'ic_spearman', as_of)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retours et santé d'un modèle (un run du flow).")
    parser.add_argument(
        '--model',
        choices=(MODEL_NAME, RETURN_MODEL_NAME),
        default=MODEL_NAME,
        help='modèle à évaluer',
    )
    parser.add_argument(
        '--as-of',
        type=datetime.fromisoformat,
        default=None,
        help="instant d'observation, avec fuseau (défaut : maintenant)",
    )
    args = parser.parse_args(argv)
    feedback = feedback_returns if args.model == RETURN_MODEL_NAME else feedback_volatility
    print(feedback(as_of=args.as_of))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
