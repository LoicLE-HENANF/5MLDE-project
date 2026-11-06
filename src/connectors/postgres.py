"""Connecteur PostgreSQL : écriture des traces (base mlops, schéma tracing, rôle api_writer).

Grafana lit ces tables (datasource projet-traces) : c'est par ici que passe tout ce qui doit
apparaître sur les dashboards. Les traces sont en ajout seul (INSERT, jamais UPDATE/DELETE).

    with PostgresConnector(PostgresConfig.from_env()) as db:   # ouvre un pool (API, worker)
        db.record_pipeline_run(...)

Sans `with`, chaque appel ouvre et ferme sa propre connexion (tâche ponctuelle).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
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
}
RUN_STATUSES = ('success', 'failure', 'cancelled')
MODEL_EVENTS = ('registered', 'promoted', 'retired')
APPLICATION_NAME = '5mlde-projet'


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
    def insert(self, table: str, row: Mapping[str, Any]) -> int:
        """INSERT d'une ligne dans tracing.<table> ; renvoie l'id généré."""
        allowed = TABLES.get(table)
        if allowed is None:
            raise ValueError(f'table inconnue : {table!r} (connues : {sorted(TABLES)})')
        unknown = set(row) - allowed
        if unknown or not row:
            raise ValueError(f'colonnes invalides pour {table} : {sorted(unknown) or "aucune"}')
        columns = list(row)
        query = sql.SQL('INSERT INTO {}.{} ({}) VALUES ({}) RETURNING id').format(
            sql.Identifier(SCHEMA),
            sql.Identifier(table),
            sql.SQL(', ').join(map(sql.Identifier, columns)),
            sql.SQL(', ').join(sql.Placeholder() * len(columns)),
        )
        values = [Jsonb(v) if isinstance(v, (dict, list)) else v for v in row.values()]
        with self.connection() as conn:
            return conn.execute(query, values).fetchone()[0]

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
