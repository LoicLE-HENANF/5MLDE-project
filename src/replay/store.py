"""Traces du rejeu dans le schéma `tracing`, toutes marquées `mode = 'replay'`.

Les écritures passent par le rôle applicatif (api_writer : SELECT/INSERT) et sont idempotentes
(contraintes d'unicité, ON CONFLICT DO NOTHING) : un mois interrompu peut être rejoué sans doublon.
Seule la remise à zéro (`reset`) supprime des lignes, avec le compte administrateur PostgreSQL.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from connectors.postgres import PostgresConnector

MODE = 'replay'
PIPELINE_PREFIX = 'replay-'

PREDICTION_SQL = """
INSERT INTO tracing.predictions
    (model_name, version, alias, symbol, open_time, horizon_hours, prediction, baseline, position, mode,
     mlflow_run_id)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'replay', %s)
ON CONFLICT DO NOTHING
"""

FEEDBACK_SQL = """
INSERT INTO tracing.feedback (prediction_id, realized, loss, baseline_loss)
VALUES (%s, %s, %s, %s)
ON CONFLICT DO NOTHING
"""

PENDING_SQL = """
SELECT p.id AS prediction_id, p.symbol, p.open_time, p.horizon_hours, p.prediction, p.baseline
FROM tracing.predictions p
LEFT JOIN tracing.feedback f ON f.prediction_id = p.id
WHERE f.id IS NULL AND p.mode = 'replay' AND p.model_name = %s AND p.open_time >= %s AND p.open_time < %s
ORDER BY p.open_time
"""

SCORED_SQL = """
SELECT p.model_name, p.version, p.alias, p.mode, p.symbol, p.open_time, p.position,
       p.prediction, p.baseline, f.realized, f.loss, f.baseline_loss
FROM tracing.predictions p
JOIN tracing.feedback f ON f.prediction_id = p.id
WHERE p.mode = 'replay' AND p.model_name = %s AND p.open_time >= %s AND p.open_time < %s
"""

HEALTH_SQL = """
SELECT day, degraded FROM tracing.model_health
WHERE mode = 'replay' AND model_name = %s AND version = %s AND metric = %s AND day <= %s
ORDER BY day DESC
LIMIT %s
"""

DONE_SQL = """
SELECT details->>'mois' AS month FROM tracing.pipeline_runs
WHERE pipeline = %s AND status = 'success'
"""

RESET_SQL = (
    'DELETE FROM tracing.feedback f USING tracing.predictions p '
    "WHERE f.prediction_id = p.id AND p.mode = 'replay'",
    "DELETE FROM tracing.predictions WHERE mode = 'replay'",
    "DELETE FROM tracing.model_health WHERE mode = 'replay'",
    "DELETE FROM tracing.model_events WHERE model_name LIKE '%-replay'",
    "DELETE FROM tracing.pipeline_runs WHERE pipeline LIKE 'replay-%'",
)


def _optional(value: Any) -> float | None:
    return None if value is None or not np.isfinite(float(value)) else float(value)


class ReplayStore:
    """Lectures et écritures du rejeu ; `db` est un `PostgresConnector` ouvert (rôle api_writer)."""

    def __init__(self, db: PostgresConnector) -> None:
        self.db = db

    # -- reprise ---------------------------------------------------------------------------------
    def done_months(self, model_name: str) -> set[str]:
        return {row['month'] for row in self.db.select(DONE_SQL, [PIPELINE_PREFIX + model_name])}

    def mark_month(
        self, model_name: str, month: str, started: datetime, ended: datetime, details: Mapping
    ) -> None:
        self.db.record_pipeline_run(
            pipeline=PIPELINE_PREFIX + model_name,
            status='success',
            started_at=started,
            ended_at=ended,
            details={'mode': MODE, 'mois': month, **details},
        )

    # -- écritures -------------------------------------------------------------------------------
    def write_predictions(
        self,
        model_name: str,
        version: int,
        alias: str,
        horizon_hours: int,
        run_id: str | None,
        rows: pd.DataFrame,
    ) -> int:
        values = [
            (
                model_name,
                int(version),
                alias,
                row.symbol,
                row.open_time.to_pydatetime(),
                int(horizon_hours),
                float(row.prediction),
                _optional(row.baseline),
                _optional(row.position),
                run_id,
            )
            for row in rows.itertuples(index=False)
            if np.isfinite(float(row.prediction))
        ]
        return self._many(PREDICTION_SQL, values)

    def write_feedback(self, scored: pd.DataFrame) -> int:
        values = [
            (int(row.prediction_id), float(row.realized), float(row.loss), _optional(row.baseline_loss))
            for row in scored.itertuples(index=False)
        ]
        return self._many(FEEDBACK_SQL, values)

    def write_health(self, rows: Iterable[Mapping[str, Any]]) -> int:
        written = 0
        for row in rows:
            written += self.db.record_model_health(**row) is not None
        return written

    def record_event(self, **event: Any) -> Any:
        return self.db.record_model_event(**event)

    def _many(self, query: str, values: Sequence[tuple]) -> int:
        if not values:
            return 0
        with self.db.connection() as conn, conn.cursor() as cursor:
            cursor.executemany(query, values)
            return cursor.rowcount if cursor.rowcount >= 0 else len(values)

    # -- lectures --------------------------------------------------------------------------------
    def pending(self, model_name: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        return self.db.select(PENDING_SQL, [model_name, start, end])

    def scored(self, model_name: str, start: datetime, end: datetime) -> pd.DataFrame:
        frame = pd.DataFrame(self.db.select(SCORED_SQL, [model_name, start, end]))
        if not frame.empty:
            frame['open_time'] = pd.to_datetime(frame['open_time'], utc=True)
        return frame

    def health_history(
        self, model_name: str, version: int, metric: str, until: date, limit: int
    ) -> list[dict]:
        return self.db.select(HEALTH_SQL, [model_name, int(version), metric, until, limit])


def reset(admin: PostgresConnector) -> None:
    """Supprime toutes les traces `replay` (compte administrateur : api_writer n'a pas DELETE)."""
    with admin.connection() as conn:
        for statement in RESET_SQL:
            conn.execute(statement)


__all__ = ['MODE', 'ReplayStore', 'reset']
