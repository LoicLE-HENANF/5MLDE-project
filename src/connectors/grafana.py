"""Connecteur Grafana (API HTTP) : santé, test des datasources, dashboards, annotations.

Grafana ne reçoit pas de données de l'application : il lit PostgreSQL (traces, Prefect, MLflow).
Ce connecteur sert à vérifier que cette chaîne de lecture fonctionne, à publier des dashboards
générés par code et à poser des annotations (déploiement, promotion de modèle...).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

import httpx

from connectors.config import GrafanaConfig
from connectors.errors import ConnectorError


class GrafanaConnector:
    service = 'Grafana'

    def __init__(self, config: GrafanaConfig, client: httpx.Client | None = None) -> None:
        self.config = config
        if client is None:
            headers = {'Authorization': f'Bearer {config.token}'} if config.token else {}
            auth = None if config.token else (config.user, config.password)
            client = httpx.Client(base_url=config.url, auth=auth, headers=headers, timeout=10.0)
        self.client = client

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> GrafanaConnector:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _request(
        self, method: str, path: str, *, accept: Iterable[int] = (), **kwargs: Any
    ) -> httpx.Response:
        target = f'{self.config.url}{path}'
        try:
            response = self.client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ConnectorError(
                self.service,
                target,
                f'injoignable ({type(exc).__name__})',
                'vérifier GRAFANA_URL et que le service grafana de compose est healthy',
            ) from exc
        if response.status_code in (401, 403):
            raise ConnectorError(
                self.service,
                target,
                f'identifiants refusés (HTTP {response.status_code})',
                'vérifier GRAFANA_TOKEN ou GRAFANA_ADMIN_USER/GRAFANA_ADMIN_PASSWORD',
            )
        if response.status_code >= 400 and response.status_code not in set(accept):
            raise ConnectorError(self.service, target, f'HTTP {response.status_code} : {response.text[:200]}')
        return response

    # -- santé ---------------------------------------------------------------------------------
    def check(self) -> None:
        """Grafana répond, sa base interne est OK et chaque datasource joint PostgreSQL."""
        health = self._request('GET', '/api/health').json()
        if health.get('database') != 'ok':
            raise ConnectorError(self.service, self.config.url, f'base interne KO : {health}')
        for uid in self.config.datasources:
            path = f'/api/datasources/uid/{uid}/health'
            response = self._request('GET', path, accept=(400, 404))
            if response.status_code == 404:
                raise ConnectorError(
                    self.service,
                    f'{self.config.url}{path}',
                    f'datasource {uid!r} absente',
                    'vérifier infra/grafana/provisioning/datasources et GRAFANA_DATASOURCE_UIDS',
                )
            body = response.json()
            if response.status_code != 200 or body.get('status') != 'OK':
                raise ConnectorError(
                    self.service,
                    f'{self.config.url}{path}',
                    f'datasource {uid!r} KO : {body.get("message", body)}',
                    'vérifier GRAFANA_DB_PASSWORD et les droits de grafana_reader',
                )

    # -- écritures -----------------------------------------------------------------------------
    def upsert_dashboard(
        self, dashboard: Mapping[str, Any], *, folder_uid: str | None = None, message: str | None = None
    ) -> str:
        """Crée ou remplace un dashboard (par uid) ; renvoie son URL complète."""
        if not dashboard.get('uid') or not dashboard.get('title'):
            raise ValueError('le dashboard doit avoir un uid et un title')
        payload: dict[str, Any] = {'dashboard': {**dashboard, 'id': None}, 'overwrite': True}
        if folder_uid:
            payload['folderUid'] = folder_uid
        if message:
            payload['message'] = message
        body = self._request('POST', '/api/dashboards/db', json=payload).json()
        return f'{self.config.url}{body["url"]}'

    def annotate(
        self,
        text: str,
        *,
        tags: Iterable[str] = (),
        when: datetime | None = None,
        dashboard_uid: str | None = None,
    ) -> int:
        """Annotation (globale ou sur un dashboard), visible sur les séries temporelles."""
        payload: dict[str, Any] = {'text': text, 'tags': list(tags)}
        if when is not None:
            if when.tzinfo is None:
                raise ValueError('when doit être daté avec un fuseau (timezone-aware)')
            payload['time'] = int(when.timestamp() * 1000)
        if dashboard_uid:
            payload['dashboardUID'] = dashboard_uid
        return int(self._request('POST', '/api/annotations', json=payload).json()['id'])
