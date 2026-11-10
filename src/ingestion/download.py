"""Client HTTP des deux sources Binance.

Archives : le `.CHECKSUM` est téléchargé d'abord ; son absence (404) signifie que le mois n'est pas
encore publié. Le zip n'est lu qu'après vérification de son SHA-256.

Les erreurs réseau, 429 et 5xx sont retentées avec un délai croissant (en respectant `Retry-After`).
Un checksum faux, un 418 (adresse bannie par Binance) ou un autre code HTTP ne le sont jamais.
"""

from __future__ import annotations

import hashlib
import io
import time
import zipfile
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

import httpx

from marketdata import sources
from marketdata.sources import SourceFormatError
from marketdata.universe import INTERVAL_DELTA

_RETRIED_STATUSES = frozenset({429, 500, 502, 503, 504})
_INTERVAL_MS = int(INTERVAL_DELTA.total_seconds() * 1000)


class DownloadError(RuntimeError):
    """Source injoignable ou réponse HTTP inattendue (après les tentatives prévues)."""


class ChecksumMismatch(SourceFormatError):
    """Le zip téléchargé ne correspond pas à son `.CHECKSUM`."""


def _to_ms(moment: datetime) -> int:
    if moment.tzinfo is None:
        raise ValueError(f'datetime sans fuseau : {moment!r} (UTC attendu)')
    return int(moment.timestamp() * 1000)


class BinanceClient:
    def __init__(
        self,
        http: httpx.Client | None = None,
        *,
        retries: int = 3,
        backoff_s: float = 1.0,
        timeout_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = http
        self._owned = http is None
        self.retries = retries
        self.backoff_s = backoff_s
        self.timeout_s = timeout_s
        self._sleep = sleep

    @property
    def http(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=self.timeout_s, follow_redirects=True)
        return self._http

    def close(self) -> None:
        if self._owned and self._http is not None:
            self._http.close()
            self._http = None

    def __enter__(self) -> BinanceClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response | None:
        """Réponse 200, ou None sur 404 ; DownloadError sinon."""
        problem = ''
        for attempt in range(self.retries + 1):
            delay = self.backoff_s * 2**attempt
            try:
                response = self.http.get(url, params=params, timeout=self.timeout_s)
            except httpx.TransportError as exc:
                problem = f'{type(exc).__name__}'
            else:
                if response.status_code == 200:
                    return response
                if response.status_code == 404:
                    return None
                if response.status_code not in _RETRIED_STATUSES:
                    raise DownloadError(f'{url} : HTTP {response.status_code} {response.text[:200]!r}')
                problem = f'HTTP {response.status_code}'
                retry_after = response.headers.get('Retry-After', '')
                if retry_after.isdigit():
                    delay = max(delay, float(retry_after))
            if attempt < self.retries:
                self._sleep(delay)
        raise DownloadError(f'{url} : {problem} après {self.retries + 1} tentative(s)')

    # -- archives mensuelles -------------------------------------------------------------------
    def archive_rows(self, symbol: str, month: date) -> tuple[list[list[str]], str] | None:
        """(lignes brutes du mois, SHA-256 vérifié du zip), ou None si le mois n'est pas publié."""
        checksum = self._get(sources.checksum_url(symbol, month))
        if checksum is None:
            return None
        expected = sources.parse_checksum(checksum.text)
        url = sources.archive_url(symbol, month)
        archive = self._get(url)
        if archive is None:
            raise DownloadError(f'{url} : .CHECKSUM publié mais archive absente (HTTP 404)')
        actual = hashlib.sha256(archive.content).hexdigest()
        if actual != expected:
            raise ChecksumMismatch(f'{url} : SHA-256 {actual} au lieu de {expected}')
        try:
            with zipfile.ZipFile(io.BytesIO(archive.content)) as zf:
                names = zf.namelist()
                if len(names) != 1:
                    raise SourceFormatError(f'{url} : 1 fichier attendu dans le zip, {len(names)} trouvé(s)')
                content = zf.read(names[0])
        except zipfile.BadZipFile as exc:
            raise SourceFormatError(f'{url} : zip illisible ({exc})') from exc
        return sources.read_archive_csv(content), expected

    # -- API REST ------------------------------------------------------------------------------
    def recent_rows(self, symbol: str, start: datetime, end: datetime) -> list[list[Any]]:
        """Bougies dont l'ouverture est dans [start, end) (pages de 1000 au plus)."""
        start_ms, end_ms = _to_ms(start), _to_ms(end) - 1
        rows: list[list[Any]] = []
        while start_ms <= end_ms:
            response = self._get(sources.REST_KLINES_URL, sources.rest_params(symbol, start_ms, end_ms))
            if response is None:
                raise DownloadError(f'{sources.REST_KLINES_URL} : HTTP 404')
            page = response.json()
            if not isinstance(page, list):
                raise SourceFormatError(f'{symbol} : réponse REST inattendue {str(page)[:200]!r}')
            if not page:
                break
            rows.extend(page)
            start_ms = int(page[-1][0]) + _INTERVAL_MS
            if len(page) < sources.REST_MAX_LIMIT:
                break
        return rows
