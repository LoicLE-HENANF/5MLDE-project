"""API FastAPI du service.

Les prédictions sont produites par le flow `predict-volatility` et lues ici dans
`tracing.predictions` : l'API sert la dernière prévision de chaque symbole, elle ne recalcule rien.
Les actions (marche / arrêt, promotion, retour arrière) sont validées puis tracées en base ;
l'application Streamlit passe uniquement par ces endpoints.

    uv run uvicorn api.app:create_app --factory --port 8000
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from psycopg import sql
from pydantic import BaseModel, Field

from connectors.errors import ConnectorError
from connectors.postgres import SCHEMA, PostgresConnector
from registry import CHAMPION, RegistryError, promote, rollback, status

MODELS = ('volatilite', 'rendement')


@dataclass
class Services:
    """Connecteurs utilisés par l'API (remplaçables dans les tests)."""

    db: PostgresConnector
    mlflow: Any  # connectors.mlflow.MlflowConnector


class SystemChange(BaseModel):
    active: bool
    changed_by: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=500)


class Promotion(BaseModel):
    version: int = Field(gt=0)
    changed_by: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=400)


def _table(name: str) -> sql.Composable:
    return sql.SQL('{}.{}').format(sql.Identifier(SCHEMA), sql.Identifier(name))


LATEST_PREDICTIONS = sql.SQL(
    'SELECT DISTINCT ON (symbol) id, symbol, open_time, horizon_hours, prediction, baseline, position, '
    'version, alias, mlflow_run_id, recorded_at FROM {} '
    "WHERE model_name = %s AND alias = %s AND mode = 'live' "
    'ORDER BY symbol, open_time DESC, version DESC'
).format(_table('predictions'))
HEALTH = sql.SQL(
    'SELECT day, version, mode, samples, metric, value, baseline_value, pnl, baseline_pnl, degraded '
    "FROM {} WHERE model_name = %s AND mode = 'live' AND day > current_date - %s "
    'ORDER BY day DESC, version DESC'
).format(_table('model_health'))
EVENTS = sql.SQL(
    'SELECT version, event, alias, details, recorded_at FROM {} '
    'WHERE model_name = %s ORDER BY id DESC LIMIT %s'
).format(_table('model_events'))
SYSTEM_HISTORY = sql.SQL(
    'SELECT active, previous, changed_by, reason, recorded_at FROM {} ORDER BY id DESC LIMIT %s'
).format(_table('system_changes'))


def _known(name: str) -> str:
    if name not in MODELS:
        raise HTTPException(404, f'modèle inconnu : {name!r} (connus : {", ".join(MODELS)})')
    return name


def _probe(check: Callable[[], None]) -> str:
    try:
        check()
    except ConnectorError as exc:
        return f'erreur : {exc}'
    return 'ok'


def default_services() -> Services:
    """Connecteurs réels configurés par l'environnement (`.env`)."""
    from connectors.config import MlflowConfig, PostgresConfig, load_env
    from connectors.mlflow import MlflowConnector

    env = load_env()
    db = PostgresConnector(PostgresConfig.from_env(env))
    db.open()
    return Services(db=db, mlflow=MlflowConnector(MlflowConfig.from_env(env)))


def create_app(services: Callable[[], Services] = default_services) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = services()
        try:
            yield
        finally:
            app.state.services.db.close()

    app = FastAPI(title='5mlde-project', summary=__doc__.split('\n')[0], lifespan=lifespan)

    def svc() -> Services:
        return app.state.services

    @app.get('/health')
    def health() -> JSONResponse:
        """Santé des dépendances, état du système et alias en service ; 503 si une dépendance manque."""
        s = svc()
        checks = {'postgres': _probe(s.db.check), 'mlflow': _probe(s.mlflow.check)}
        body: dict[str, Any] = {'services': checks}
        if checks['postgres'] == 'ok':
            body['system'] = s.db.system_state()
        if checks['mlflow'] == 'ok':
            body['models'] = {name: status(s.mlflow, name) for name in MODELS}
        healthy = all(v == 'ok' for v in checks.values())
        body['status'] = 'ok' if healthy else 'degraded'
        return JSONResponse(jsonable_encoder(body), status_code=200 if healthy else 503)

    @app.get('/predictions/latest')
    def latest_predictions(
        model: str = 'volatilite', alias: Literal['champion', 'challenger'] = 'champion'
    ) -> dict[str, Any]:
        """Dernière prévision de chaque symbole pour l'alias demandé."""
        rows = svc().db.select(LATEST_PREDICTIONS, (_known(model), alias))
        return {'model': model, 'alias': alias, 'predictions': rows}

    @app.get('/models/{name}')
    def model(name: str, days: int = Query(14, ge=1, le=90)) -> dict[str, Any]:
        """Alias du registre, santé (niveau 3) des derniers jours et derniers événements du registre."""
        s = svc()
        _known(name)
        return {
            'model': name,
            'aliases': status(s.mlflow, name),
            'health': s.db.select(HEALTH, (name, days)),
            'events': s.db.select(EVENTS, (name, 20)),
        }

    @app.post('/models/{name}/promote')
    def promote_version(name: str, body: Promotion) -> dict[str, Any]:
        """Promotion manuelle en champion ; l'ancien champion devient `precedent`."""
        s = svc()
        _known(name)
        try:
            version = promote(
                s.mlflow,
                name,
                body.version,
                reason=f'{body.reason} (par {body.changed_by})',
                record=s.db.record_model_event,
            )
        except RegistryError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {'model': name, CHAMPION: str(version.version), 'aliases': status(s.mlflow, name)}

    @app.post('/models/{name}/rollback')
    def rollback_version(name: str) -> dict[str, Any]:
        """Retour arrière : la version `precedent` redevient champion."""
        s = svc()
        _known(name)
        try:
            version = rollback(s.mlflow, name, record=s.db.record_model_event)
        except RegistryError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {'model': name, CHAMPION: str(version.version), 'aliases': status(s.mlflow, name)}

    @app.get('/system/state')
    def system_state(history: int = Query(10, ge=0, le=100)) -> dict[str, Any]:
        """État courant (actif tant que personne ne l'a arrêté) et derniers changements."""
        db = svc().db
        return {
            'state': db.system_state(),
            'history': db.select(SYSTEM_HISTORY, (history,)) if history else [],
        }

    @app.post('/system/state')
    def change_system_state(body: SystemChange) -> dict[str, Any]:
        """Marche / arrêt : les flows lisent cet état au démarrage et s'arrêtent proprement."""
        db = svc().db
        change_id = db.record_system_change(
            active=body.active, changed_by=body.changed_by, reason=body.reason
        )
        return {'id': change_id, 'state': db.system_state()}

    return app
