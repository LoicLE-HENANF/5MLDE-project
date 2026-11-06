"""Appel HTTP de santé partagé (MLflow, Prefect) : traduit les erreurs réseau en ConnectorError."""

from __future__ import annotations

import httpx

from connectors.errors import ConnectorError


def check_http(
    service: str,
    url: str,
    *,
    timeout: float = 5.0,
    client: httpx.Client | None = None,
    hint: str | None = None,
) -> httpx.Response:
    try:
        response = (client or httpx).get(url, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise ConnectorError(service, url, f'délai dépassé ({timeout} s)', hint) from exc
    except httpx.HTTPError as exc:
        raise ConnectorError(service, url, f'injoignable ({type(exc).__name__})', hint) from exc
    if response.status_code != 200:
        raise ConnectorError(service, url, f'HTTP {response.status_code} sur la route de santé', hint)
    return response
