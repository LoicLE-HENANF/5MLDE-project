"""Client Binance : transport HTTP simulé (aucun réseau)."""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import UTC, date, datetime, timedelta

import httpx
import pytest

from ingestion.download import BinanceClient, ChecksumMismatch, DownloadError
from marketdata import sources
from marketdata.sources import SourceFormatError
from tests.conftest import refuse

HOUR_MS = 3_600_000
T0 = datetime(2025, 1, 1, tzinfo=UTC)
T0_MS = int(T0.timestamp() * 1000)
CSV = b'1735689600000,1,2,0.5,1.5,10,1735693199999,15,3,4,6,0\n'


def zipped(content: bytes = CSV, names: tuple[str, ...] = ('BTCUSDT-1h-2024-12.csv',)) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as zf:
        for name in names:
            zf.writestr(name, content)
    return buffer.getvalue()


def checksum_of(payload: bytes) -> str:
    return f'{hashlib.sha256(payload).hexdigest()}  BTCUSDT-1h-2024-12.zip\n'


def client_for(http_client, handler, **kwargs) -> tuple[BinanceClient, list[float]]:
    sleeps: list[float] = []
    client = BinanceClient(http_client(handler), sleep=sleeps.append, **kwargs)
    return client, sleeps


def archive_handler(payload: bytes, checksum: str | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith('.CHECKSUM'):
            return httpx.Response(200, text=checksum or checksum_of(payload))
        return httpx.Response(200, content=payload)

    return handler


# -- archives ----------------------------------------------------------------------------------
def test_archive_verified_and_read(http_client):
    payload = zipped()
    client, _ = client_for(http_client, archive_handler(payload))
    rows, digest = client.archive_rows('BTCUSDT', date(2024, 12, 1))
    assert digest == hashlib.sha256(payload).hexdigest()
    assert rows == [CSV.decode().strip().split(',')]


def test_archive_checksum_requested_first(http_client):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(404)

    client, _ = client_for(http_client, handler)
    assert client.archive_rows('BTCUSDT', date(2026, 11, 1)) is None
    assert seen == [sources.checksum_url('BTCUSDT', date(2026, 11, 1))]


def test_archive_checksum_mismatch(http_client):
    client, sleeps = client_for(http_client, archive_handler(zipped(), checksum_of(b'autre')))
    with pytest.raises(ChecksumMismatch, match='SHA-256'):
        client.archive_rows('BTCUSDT', date(2024, 12, 1))
    assert sleeps == []  # jamais retenté


def test_archive_missing_after_checksum(http_client):
    def handler(request):
        if str(request.url).endswith('.CHECKSUM'):
            return httpx.Response(200, text=checksum_of(b'x'))
        return httpx.Response(404)

    client, _ = client_for(http_client, handler)
    with pytest.raises(DownloadError, match='archive absente'):
        client.archive_rows('BTCUSDT', date(2024, 12, 1))


@pytest.mark.parametrize(
    ('payload', 'match'),
    [(b'pas un zip', 'zip illisible'), (zipped(names=('a.csv', 'b.csv')), '1 fichier attendu')],
)
def test_archive_bad_zip(http_client, payload, match):
    client, _ = client_for(http_client, archive_handler(payload))
    with pytest.raises(SourceFormatError, match=match):
        client.archive_rows('BTCUSDT', date(2024, 12, 1))


# -- tentatives --------------------------------------------------------------------------------
def test_retries_transient_errors_with_backoff(http_client):
    answers = [httpx.Response(503), httpx.Response(429, headers={'Retry-After': '7'}), httpx.Response(404)]
    client, sleeps = client_for(http_client, lambda r: answers.pop(0), backoff_s=1.0)
    assert client.archive_rows('BTCUSDT', date(2024, 12, 1)) is None
    assert sleeps == [1.0, 7.0]


def test_gives_up_after_retries(http_client):
    client, sleeps = client_for(http_client, refuse, retries=2)
    with pytest.raises(DownloadError, match='ConnectError après 3 tentative'):
        client.archive_rows('BTCUSDT', date(2024, 12, 1))
    assert len(sleeps) == 2


@pytest.mark.parametrize('status', [400, 403, 418])
def test_non_transient_status_not_retried(http_client, status):
    client, sleeps = client_for(http_client, lambda r: httpx.Response(status, text='banned'))
    with pytest.raises(DownloadError, match=f'HTTP {status}'):
        client.archive_rows('BTCUSDT', date(2024, 12, 1))
    assert sleeps == []


# -- REST --------------------------------------------------------------------------------------
def kline(open_ms: int) -> list:
    return [open_ms, '1', '2', '0.5', '1.5', '10', open_ms + HOUR_MS - 1, '15', 3, '4', '6', '0']


def test_recent_paginates(http_client, monkeypatch):
    monkeypatch.setattr(sources, 'REST_MAX_LIMIT', 2)
    calls = []

    def handler(request):
        params = dict(request.url.params)
        calls.append(params)
        start, end = int(params['startTime']), int(params['endTime'])
        opens = [ms for ms in range(start, end + 1, HOUR_MS)][:2]
        return httpx.Response(200, json=[kline(ms) for ms in opens])

    client, _ = client_for(http_client, handler)
    rows = client.recent_rows('BTCUSDT', T0, T0 + timedelta(hours=5))
    assert [r[0] for r in rows] == [T0_MS + i * HOUR_MS for i in range(5)]
    assert [int(c['startTime']) for c in calls] == [T0_MS, T0_MS + 2 * HOUR_MS, T0_MS + 4 * HOUR_MS]
    assert all(int(c['endTime']) == T0_MS + 5 * HOUR_MS - 1 for c in calls)


def test_recent_empty(http_client):
    client, _ = client_for(http_client, lambda r: httpx.Response(200, json=[]))
    assert client.recent_rows('BTCUSDT', T0, T0 + timedelta(hours=2)) == []


def test_recent_unexpected_payload(http_client):
    client, _ = client_for(http_client, lambda r: httpx.Response(200, json={'code': -1121}))
    with pytest.raises(SourceFormatError, match='réponse REST inattendue'):
        client.recent_rows('BTCUSDT', T0, T0 + timedelta(hours=2))


def test_recent_requires_timezone(http_client):
    client, _ = client_for(http_client, lambda r: httpx.Response(200, json=[]))
    with pytest.raises(ValueError, match='sans fuseau'):
        client.recent_rows('BTCUSDT', datetime(2025, 1, 1), T0)


def test_owned_client_closed():
    client = BinanceClient()
    http = client.http
    client.close()
    assert http.is_closed
