"""Connecteur Prefect : santé du serveur, déclenchement de déploiements, liens run <-> MLflow.

Production : les traitements tournent comme déploiements Prefect (serve()/worker) et sont
déclenchés par planning ou par trigger() depuis l'API/l'application, plus par des commandes
lancées à la main. Les helpers de contexte fonctionnent dans et hors d'un run (tests, CLI).

La bibliothèque Prefect lit elle-même PREFECT_API_URL (environnement, puis .env du dossier
courant). check() vérifie que cette valeur est la même que celle de notre configuration :
sinon un flow tournerait sur un serveur éphémère local, sans erreur visible.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from uuid import UUID

import httpx

from connectors._http import check_http
from connectors.config import PrefectConfig
from connectors.errors import ConfigError

_FALLBACK_LOGGER = logging.getLogger('connectors.prefect')


def _settings_api_url() -> str | None:
    from prefect.settings import get_current_settings

    url = get_current_settings().api.url
    return str(url).rstrip('/') if url else None


class PrefectConnector:
    service = 'Prefect'

    def __init__(self, config: PrefectConfig, http: httpx.Client | None = None) -> None:
        self.config = config
        self._http = http

    def check(self, timeout: float = 5.0) -> None:
        library_url = _settings_api_url()
        if library_url != self.config.api_url:
            raise ConfigError(
                f'Prefect utiliserait {library_url or "un serveur éphémère local"} au lieu de '
                f'{self.config.api_url} : exporter PREFECT_API_URL ou lancer depuis la racine du projet'
            )
        check_http(
            self.service,
            f'{self.config.api_url}/health',
            timeout=timeout,
            client=self._http,
            hint='vérifier PREFECT_API_URL et que le service prefect de compose est healthy',
        )

    def flow_run_url(self, flow_run_id: str | UUID) -> str:
        return f'{self.config.ui_url}/runs/flow-run/{flow_run_id}'

    def trigger(
        self,
        deployment: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        wait_s: float = 0,
        tags: list[str] | None = None,
    ) -> Any:
        """Crée un run du déploiement 'flow/deploiement' ; wait_s=0 : n'attend pas sa fin."""
        from prefect.deployments import run_deployment

        return run_deployment(name=deployment, parameters=dict(parameters or {}), timeout=wait_s, tags=tags)


# -- helpers de contexte (sans configuration) ----------------------------------------------------
def run_context() -> dict[str, str]:
    """Identifiants du flow run / task run courant ; {} hors Prefect."""
    from prefect.context import FlowRunContext, TaskRunContext

    ids: dict[str, str] = {}
    flow_ctx = FlowRunContext.get()
    if flow_ctx is not None and flow_ctx.flow_run is not None:
        ids['flow_run_id'] = str(flow_ctx.flow_run.id)
        ids['flow_run_name'] = flow_ctx.flow_run.name
        ids['flow_name'] = flow_ctx.flow.name
    task_ctx = TaskRunContext.get()
    if task_ctx is not None:
        ids['task_run_id'] = str(task_ctx.task_run.id)
        ids.setdefault('flow_run_id', str(task_ctx.task_run.flow_run_id))
    return ids


def get_logger() -> logging.Logger | logging.LoggerAdapter:
    """Logger du run (visible dans l'UI Prefect), logger Python standard hors run."""
    from prefect.exceptions import MissingContextError
    from prefect.logging import get_run_logger

    try:
        return get_run_logger()
    except MissingContextError:
        return _FALLBACK_LOGGER


def mlflow_tags(connector: PrefectConnector | None = None) -> dict[str, str]:
    """Tags MLflow reliant un run MLflow au flow run Prefect courant ({} hors Prefect)."""
    ids = run_context()
    tags = {f'prefect.{key}': value for key, value in ids.items()}
    if connector is not None and 'flow_run_id' in ids:
        tags['prefect.flow_run_url'] = connector.flow_run_url(ids['flow_run_id'])
    return tags


def publish_markdown(markdown: str, key: str, description: str | None = None) -> UUID | None:
    """Artifact Markdown du run (onglet Artifacts). Jamais bloquant ; None hors run ou en échec."""
    if not run_context():
        return None
    try:
        from prefect.artifacts import create_markdown_artifact

        return create_markdown_artifact(markdown=markdown, key=key, description=description)
    except Exception as exc:  # noqa: BLE001 - un artifact ne doit jamais faire échouer un run
        get_logger().warning('artifact %s non publié : %s', key, exc)
        return None


def task_options(**overrides: Any) -> dict[str, Any]:
    """Options @task par défaut : pas de reprise implicite, pas de cache ni de résultat persisté."""
    from prefect.cache_policies import NO_CACHE

    return {'retries': 0, 'persist_result': False, 'cache_policy': NO_CACHE, **overrides}
