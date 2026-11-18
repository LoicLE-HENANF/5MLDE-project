"""Client HTTP de l'API du service : seul accès de l'application aux données et aux actions.

L'application ne lit ni Postgres ni MLflow : tout passe par les endpoints de `api.app`,
qui valident et tracent chaque action (marche / arrêt, promotion, retour arrière).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx

from connectors.config import load_env

DEFAULT_API_URL = 'http://localhost:8000'
MODELS = ('volatilite', 'rendement')
ALIASES = ('champion', 'challenger')


class ApiError(RuntimeError):
    """Appel refusé par l'API (message de l'API) ou API injoignable."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def api_url(env: Mapping[str, str] | None = None) -> str:
    """Adresse de l'API (`API_URL`, défaut http://localhost:8000), validée."""
    values = load_env() if env is None else env
    url = (values.get('API_URL') or DEFAULT_API_URL).strip().rstrip('/')
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.netloc:
        raise ValueError(f'API_URL={url!r} : URL http(s) attendue (ex. {DEFAULT_API_URL})')
    return url


def _detail(response: httpx.Response) -> str:
    """Message d'erreur de l'API (`detail` FastAPI), lisible même pour une erreur de validation."""
    try:
        detail = response.json().get('detail')
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, list):  # erreur de validation 422 : liste de champs refusés
        detail = ' ; '.join(f'{"/".join(map(str, d.get("loc", [])[1:]))} : {d.get("msg")}' for d in detail)
    return str(detail or response.text or response.reason_phrase)


class ApiClient:
    def __init__(self, base_url: str, *, client: httpx.Client | None = None, timeout: float = 10.0) -> None:
        self.base_url = base_url
        self._http = client or httpx.Client(base_url=base_url, timeout=timeout)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ApiClient:
        return cls(api_url(env))

    def close(self) -> None:
        self._http.close()

    def _call(self, method: str, path: str, *, accept: tuple[int, ...] = (200,), **kwargs: Any) -> Any:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(f'API injoignable ({self.base_url}) : {type(exc).__name__}') from exc
        if response.status_code not in accept:
            raise ApiError(_detail(response), response.status_code)
        return response.json()

    # -- lectures
    def health(self) -> dict[str, Any]:
        """Santé des dépendances ; la réponse 503 (dépendance manquante) reste lisible."""
        return self._call('GET', '/health', accept=(200, 503))

    def predictions(self, model: str, alias: str = 'champion') -> list[dict[str, Any]]:
        body = self._call('GET', '/predictions/latest', params={'model': model, 'alias': alias})
        return body['predictions']

    def model(self, name: str, days: int = 14) -> dict[str, Any]:
        return self._call('GET', f'/models/{name}', params={'days': days})

    def system(self, history: int = 20) -> dict[str, Any]:
        return self._call('GET', '/system/state', params={'history': history})

    # -- actions (tracées par l'API)
    def set_system(self, *, active: bool, changed_by: str, reason: str) -> dict[str, Any]:
        payload = {'active': active, 'changed_by': changed_by, 'reason': reason}
        return self._call('POST', '/system/state', json=payload)

    def promote(self, name: str, *, version: int, changed_by: str, reason: str) -> dict[str, Any]:
        payload = {'version': version, 'changed_by': changed_by, 'reason': reason}
        return self._call('POST', f'/models/{name}/promote', json=payload)

    def rollback(self, name: str) -> dict[str, Any]:
        return self._call('POST', f'/models/{name}/rollback')
