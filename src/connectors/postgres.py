"""Connecteur PostgreSQL : écriture des traces (base mlops, schéma tracing, rôle api_writer).

Grafana lit ces tables (datasource projet-traces) : c'est par ici que passe tout ce qui doit
apparaître sur les dashboards. Les traces sont en ajout seul (INSERT, jamais UPDATE/DELETE).

    with PostgresConnector(PostgresConfig.from_env()) as db:   # ouvre un pool (API, worker)
        db.record_pipeline_run(...)

Sans `with`, chaque appel ouvre et ferme sa propre connexion (tâche ponctuelle).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool, PoolTimeout

from connectors.config import PostgresConfig
from connectors.errors import ConnectorError

SCHEMA = 'tracing'
# Colonnes insérables par table : à garder alignées sur infra/postgres/init/002-tracing.sql
# (vérifié par tests/test_postgres.py).
TABLES: dict[str, frozenset[str]] = {
    'pipeline_runs': frozenset(
        {'pipeline', 'status', 'started_at', 'ended_at', 'prefect_flow_run_id', 'mlflow_run_id', 'details'}
    ),
    'model_events': frozenset(
        {'model_name', 'version', 'event', 'alias', 'mlflow_run_id', 'model_uri', 'details'}
    ),
    'ingestion_batches': frozenset(
        {
            'source',
            'symbol',
            'period_start',
            'period_end',
            'status',
            'rows',
            'source_sha256',
            'data_sha256',
            'issues',
            'prefect_flow_run_id',
        }
    ),
    # Tables du service : infra/postgres/init/004-serving.sql.
    'predictions': frozenset(
        {
            'model_name',
            'version',
            'alias',
            'symbol',
            'open_time',
            'horizon_hours',
            'prediction',
            'baseline',
            'position',
            'mode',
            'mlflow_run_id',
            'prefect_flow_run_id',
        }
    ),
    'feedback': frozenset({'prediction_id', 'realized', 'loss', 'baseline_loss', 'prefect_flow_run_id'}),
    'model_health': frozenset(
        {
            'model_name',
            'version',
            'mode',
            'day',
            'window_hours',
            'samples',
            'metric',
            'value',
            'baseline_value',
            'pnl',
            'baseline_pnl',
            'degraded',
            'details',
            'prefect_flow_run_id',
        }
    ),
    # État du système : infra/postgres/init/005-system.sql.
    'system_changes': frozenset({'active', 'previous', 'changed_by', 'reason'}),
}
RUN_STATUSES = ('success', 'failure', 'cancelled')
MODEL_EVENTS = ('registered', 'promoted', 'retired')
BATCH_SOURCES = ('archive', 'rest')
BATCH_STATUSES = ('accepted', 'rejected')
PREDICTION_ALIASES = ('champion', 'challenger')
PREDICTION_MODES = ('live', 'replay')
APPLICATION_NAME = '5mlde-project'


class PostgresConnector:
    service = 'PostgreSQL'

    def __init__(
        self,
        config: PostgresConfig,
        *,
        connect: Callable[..., psycopg.Connection] = psycopg.connect,
        pool_factory: Callable[..., ConnectionPool] = ConnectionPool,
    ) -> None:
        self.config = config
        self._connect = connect
        self._pool_factory = pool_factory
        self._pool: ConnectionPool | None = None

    @property
    def conninfo(self) -> str:
        c = self.config
        return make_conninfo(
            host=c.host,
            port=c.port,
            dbname=c.dbname,
            user=c.user,
            password=c.password,
            connect_timeout=c.timeout_s,
            application_name=APPLICATION_NAME,
        )

    def _error(self, detail: str, exc: BaseException | None = None) -> ConnectorError:
        reason = f'{detail} ({str(exc).strip().splitlines()[0]})' if exc and str(exc).strip() else detail
        return ConnectorError(
            self.service,
            self.config.target,
            reason,
            'vérifier POSTGRES_HOST/POSTGRES_PORT, API_DB_PASSWORD et que la stack tourne',
        )

    # -- santé ---------------------------------------------------------------------------------
    def check(self) -> None:
        """Connexion + présence des tables de traçabilité."""
        try:
            with self._connect(self.conninfo, autocommit=True) as conn:
                missing = [
                    table
                    for table in TABLES
                    if conn.execute('SELECT to_regclass(%s)', (f'{SCHEMA}.{table}',)).fetchone()[0] is None
                ]
        except psycopg.OperationalError as exc:
            raise self._error('connexion impossible', exc) from exc
        if missing:
            raise ConnectorError(
                self.service,
                self.config.target,
                f'tables absentes : {", ".join(missing)}',
                'volume créé avant 002-tracing.sql : docker compose down -v puis up',
            )

    # -- cycle de vie --------------------------------------------------------------------------
    def open(self) -> PostgresConnector:
        if self._pool is None:
            pool = self._pool_factory(
                self.conninfo,
                min_size=self.config.pool_min,
                max_size=self.config.pool_max,
                timeout=self.config.timeout_s,
                open=False,
                name=APPLICATION_NAME,
            )
            try:
                pool.open(wait=True, timeout=self.config.timeout_s)
            except PoolTimeout as exc:
                pool.close()
                raise self._error('pool non disponible', exc) from exc
            self._pool = pool
        return self

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def __enter__(self) -> PostgresConnector:
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection]:
        """Connexion transactionnelle : commit en sortie normale, rollback sur exception."""
        try:
            if self._pool is not None:
                with self._pool.connection() as conn:
                    yield conn
            else:
                with self._connect(self.conninfo) as conn:
                    yield conn
        except (psycopg.OperationalError, PoolTimeout) as exc:
            raise self._error('connexion perdue ou impossible', exc) from exc

    # -- écritures -----------------------------------------------------------------------------
    def insert(self, table: str, row: Mapping[str, Any], *, idempotent: bool = False) -> int | None:
        """INSERT d'une ligne dans tracing.<table> ; renvoie l'id généré.

        `idempotent` : une ligne déjà présente (contrainte d'unicité) est ignorée, l'appel renvoie None.
        """
        allowed = TABLES.get(table)
        if allowed is None:
            raise ValueError(f'table inconnue : {table!r} (connues : {sorted(TABLES)})')
        unknown = set(row) - allowed
        if unknown or not row:
            raise ValueError(f'colonnes invalides pour {table} : {sorted(unknown) or "aucune"}')
        columns = list(row)
        query = sql.SQL('INSERT INTO {}.{} ({}) VALUES ({}){} RETURNING id').format(
            sql.Identifier(SCHEMA),
            sql.Identifier(table),
            sql.SQL(', ').join(map(sql.Identifier, columns)),
            sql.SQL(', ').join(sql.Placeholder() * len(columns)),
            sql.SQL(' ON CONFLICT DO NOTHING' if idempotent else ''),
        )
        values = [Jsonb(v) if isinstance(v, (dict, list)) else v for v in row.values()]
        with self.connection() as conn:
            inserted = conn.execute(query, values).fetchone()
        return None if inserted is None else inserted[0]

    # -- lectures ------------------------------------------------------------------------------
    def select(
        self, query: str | sql.Composable, params: Sequence[Any] | Mapping[str, Any] = ()
    ) -> list[dict[str, Any]]:
        """Requête de lecture (paramètres liés, jamais interpolés) ; un dict par ligne."""
        with self.connection() as conn:
            cursor = conn.cursor(row_factory=dict_row)
            return cursor.execute(query, params).fetchall()

    def record_pipeline_run(
        self,
        *,
        pipeline: str,
        status: str,
        started_at: datetime,
        ended_at: datetime,
        prefect_flow_run_id: str | None = None,
        mlflow_run_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> int:
        if status not in RUN_STATUSES:
            raise ValueError(f'status {status!r} invalide (attendu : {RUN_STATUSES})')
        if ended_at < started_at:
            raise ValueError('ended_at est antérieur à started_at')
        return self.insert(
            'pipeline_runs',
            {
                'pipeline': pipeline,
                'status': status,
                'started_at': started_at,
                'ended_at': ended_at,
                'prefect_flow_run_id': prefect_flow_run_id,
                'mlflow_run_id': mlflow_run_id,
                'details': dict(details or {}),
            },
        )

    def record_model_event(
        self,
        *,
        model_name: str,
        version: int | str,
        event: str,
        model_uri: str,
        alias: str | None = None,
        mlflow_run_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> int:
        if event not in MODEL_EVENTS:
            raise ValueError(f'event {event!r} invalide (attendu : {MODEL_EVENTS})')
        return self.insert(
            'model_events',
            {
                'model_name': model_name,
                'version': int(version),
                'event': event,
                'alias': alias,
                'mlflow_run_id': mlflow_run_id,
                'model_uri': model_uri,
                'details': dict(details or {}),
            },
        )

    def record_ingestion_batch(
        self,
        *,
        source: str,
        symbol: str,
        period_start: datetime,
        period_end: datetime,
        status: str,
        rows: int,
        source_sha256: str | None = None,
        data_sha256: str | None = None,
        issues: list[Mapping[str, Any]] | None = None,
        prefect_flow_run_id: str | None = None,
    ) -> int:
        if source not in BATCH_SOURCES:
            raise ValueError(f'source {source!r} invalide (attendu : {BATCH_SOURCES})')
        if status not in BATCH_STATUSES:
            raise ValueError(f'status {status!r} invalide (attendu : {BATCH_STATUSES})')
        if period_end <= period_start:
            raise ValueError('period_end doit être postérieur à period_start')
        if rows < 0:
            raise ValueError('rows doit être positif ou nul')
        return self.insert(
            'ingestion_batches',
            {
                'source': source,
                'symbol': symbol,
                'period_start': period_start,
                'period_end': period_end,
                'status': status,
                'rows': rows,
                'source_sha256': source_sha256,
                'data_sha256': data_sha256,
                'issues': [dict(i) for i in issues or []],
                'prefect_flow_run_id': prefect_flow_run_id,
            },
        )

    def record_prediction(
        self,
        *,
        model_name: str,
        version: int | str,
        alias: str,
        symbol: str,
        open_time: datetime,
        horizon_hours: int,
        prediction: float,
        mode: str,
        baseline: float | None = None,
        position: float | None = None,
        mlflow_run_id: str | None = None,
        prefect_flow_run_id: str | None = None,
    ) -> int | None:
        """Trace une prédiction ; None si elle existait déjà (inférence idempotente)."""
        if alias not in PREDICTION_ALIASES:
            raise ValueError(f'alias {alias!r} invalide (attendu : {PREDICTION_ALIASES})')
        if mode not in PREDICTION_MODES:
            raise ValueError(f'mode {mode!r} invalide (attendu : {PREDICTION_MODES})')
        if alias != 'champion' and position is not None:
            raise ValueError('seul le champion porte une position')
        if horizon_hours <= 0:
            raise ValueError('horizon_hours doit être strictement positif')
        if not math.isfinite(float(prediction)):
            raise ValueError('prédiction non finie')
        return self.insert(
            'predictions',
            {
                'model_name': model_name,
                'version': int(version),
                'alias': alias,
                'symbol': symbol,
                'open_time': open_time,
                'horizon_hours': int(horizon_hours),
                'prediction': float(prediction),
                'baseline': _optional(baseline),
                'position': _optional(position),
                'mode': mode,
                'mlflow_run_id': mlflow_run_id,
                'prefect_flow_run_id': prefect_flow_run_id,
            },
            idempotent=True,
        )

    def record_feedback(
        self,
        *,
        prediction_id: int,
        realized: float,
        loss: float | None = None,
        baseline_loss: float | None = None,
        prefect_flow_run_id: str | None = None,
    ) -> int | None:
        """Trace la valeur réalisée d'une prédiction (une seule fois par prédiction)."""
        if not math.isfinite(float(realized)):
            raise ValueError('valeur réalisée non finie')
        return self.insert(
            'feedback',
            {
                'prediction_id': int(prediction_id),
                'realized': float(realized),
                'loss': _optional(loss),
                'baseline_loss': _optional(baseline_loss),
                'prefect_flow_run_id': prefect_flow_run_id,
            },
            idempotent=True,
        )

    def record_model_health(
        self,
        *,
        model_name: str,
        version: int | str,
        mode: str,
        day: date,
        window_hours: int,
        samples: int,
        metric: str,
        degraded: bool,
        value: float | None = None,
        baseline_value: float | None = None,
        pnl: float | None = None,
        baseline_pnl: float | None = None,
        details: Mapping[str, Any] | None = None,
        prefect_flow_run_id: str | None = None,
    ) -> int | None:
        """Trace la santé (niveau 3) d'une version pour un jour ; une seule ligne par jour."""
        if mode not in PREDICTION_MODES:
            raise ValueError(f'mode {mode!r} invalide (attendu : {PREDICTION_MODES})')
        if samples < 0 or window_hours <= 0:
            raise ValueError('samples doit être positif ou nul et window_hours strictement positif')
        return self.insert(
            'model_health',
            {
                'model_name': model_name,
                'version': int(version),
                'mode': mode,
                'day': day,
                'window_hours': int(window_hours),
                'samples': int(samples),
                'metric': metric,
                'value': _optional(value),
                'baseline_value': _optional(baseline_value),
                'pnl': _optional(pnl),
                'baseline_pnl': _optional(baseline_pnl),
                'degraded': bool(degraded),
                'details': dict(details or {}),
                'prefect_flow_run_id': prefect_flow_run_id,
            },
            idempotent=True,
        )

    # -- état du système -------------------------------------------------------------------------
    def system_state(self) -> dict[str, Any]:
        """Dernier changement d'état ; système actif par défaut tant que personne ne l'a arrêté."""
        rows = self.select(
            sql.SQL(
                'SELECT active, changed_by, reason, recorded_at FROM {}.system_changes '
                'ORDER BY id DESC LIMIT 1'
            ).format(sql.Identifier(SCHEMA))
        )
        if not rows:
            return {'active': True, 'changed_by': None, 'reason': None, 'recorded_at': None}
        return dict(rows[0])

    def system_active(self) -> bool:
        return bool(self.system_state()['active'])

    def record_system_change(self, *, active: bool, changed_by: str, reason: str) -> int:
        """Change l'état marche / arrêt en traçant l'ancienne valeur ; renvoie l'id du changement."""
        if not isinstance(active, bool):
            raise ValueError('active doit être un booléen')
        if not changed_by.strip() or not reason.strip():
            raise ValueError('changed_by et reason sont obligatoires')
        current = self.system_state()
        previous = None if current['recorded_at'] is None else bool(current['active'])
        return self.insert(
            'system_changes',
            {
                'active': active,
                'previous': previous,
                'changed_by': changed_by.strip(),
                'reason': reason.strip(),
            },
        )


def _optional(value: float | None) -> float | None:
    """Flottant ou None (NaN et infinis compris : non représentables proprement en base)."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None
