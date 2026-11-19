"""Scénarios d'incident joués contre la stack réelle (rien n'est simulé côté contrôles).

- `lot_invalide` : le vrai flow `ingest-candles` reçoit des bougies REST dont une est corrompue
  (plus haut sous le corps de la bougie). Le contrôle qualité rejette le lot, rien n'est écrit,
  le lot est tracé `rejected` et le run Prefect finit en échec : les alertes Grafana
  « Lot d'ingestion rejeté » et « Flow Prefect en échec » se déclenchent.
- `dependance` : MLflow est arrêté, `/health` de l'API passe en 503 (dégradé), puis MLflow est
  relancé et l'API redevient saine sans redémarrage.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any
from unittest import mock

from app.client import ApiClient
from connectors.config import ROOT
from flows import ingest
from ingestion.download import BinanceClient

SYMBOL = 'BTCUSDT'
WINDOW = timedelta(hours=3)
OPEN_IDX, HIGH_IDX, CLOSE_IDX, CLOSE_TIME_IDX = 1, 2, 4, 6


def corrupt(rows: list[list[Any]], end: datetime) -> list[list[Any]]:
    """Copie des bougies où la dernière clôturée avant `end` a un plus haut sous son corps.

    La bougie en cours est écartée par le pipeline avant les contrôles : la corrompre ne
    déclencherait rien. L'incohérence OHLC est une anomalie bloquante.
    """
    end_ms = int(end.timestamp() * 1000)
    closed = [i for i, row in enumerate(rows) if int(row[CLOSE_TIME_IDX]) < end_ms]
    if not closed:
        raise RuntimeError('aucune bougie clôturée à corrompre')
    rows = [list(row) for row in rows]
    target = rows[closed[-1]]
    body_low = min(float(target[OPEN_IDX]), float(target[CLOSE_IDX]))
    target[HIGH_IDX] = f'{body_low * 0.99:.8f}'
    return rows


class CorruptedClient(BinanceClient):
    """Client Binance réel dont la réponse REST est corrompue avant les contrôles."""

    def recent_rows(self, symbol: str, start: datetime, end: datetime) -> list[list[Any]]:
        # fenêtre élargie : la reprise après la dernière bougie stockée peut être vide
        return corrupt(super().recent_rows(symbol, min(start, end - WINDOW), end), end)


def _say(message: str) -> None:
    print(message, flush=True)  # avant les sorties de docker compose


def lot_invalide(log: Callable[[str], None] = _say) -> bool:
    """Joue le flow d'ingestion avec le lot corrompu ; vrai si le lot a bien été rejeté."""
    log(f'incident lot invalide : flow {ingest.FLOW_NAME} sur {SYMBOL}, une bougie corrompue')
    with mock.patch.object(ingest, 'BinanceClient', CorruptedClient):
        try:
            result = ingest.ingest_candles(symbols=[SYMBOL], history=False)
        except ingest.IngestionFailed as exc:
            log(f'lot rejeté comme attendu :\n{exc}')
            return True
    log(f"le lot n'a pas été rejeté : {result}")
    return False


def _compose(*args: str) -> None:
    subprocess.run(['docker', 'compose', *args], cwd=ROOT, check=True)


def _health(api: ApiClient) -> dict[str, Any]:
    """Statut global et état de chaque dépendance (le détail des modèles est omis)."""
    body = api.health()
    return {'status': body.get('status'), **body.get('services', {})}


def wait_health(api: ApiClient, healthy: bool, timeout: float = 120.0, step: float = 3.0) -> dict[str, Any]:
    """Interroge `/health` jusqu'à l'état voulu (sain ou dégradé) ou l'expiration."""
    deadline = time.monotonic() + timeout
    while True:
        body = _health(api)
        if (body.get('status') == 'ok') == healthy or time.monotonic() >= deadline:
            return body
        time.sleep(step)


def dependance(
    service: str = 'mlflow',
    *,
    pause: Callable[[], None] = lambda: input('Entrée pour relancer le service... '),
    log: Callable[[str], None] = _say,
) -> bool:
    """Arrête `service`, montre l'API dégradée, relance et vérifie le retour à la normale."""
    api = ApiClient.from_env()
    try:
        log(f'santé avant : {_health(api)}')
        log(f'arrêt de {service}')
        _compose('stop', service)
        try:
            down = wait_health(api, healthy=False)
            log(f'santé pendant la panne : {down}')
            pause()
        finally:
            log(f'relance de {service}')
            _compose('start', service)
        up = wait_health(api, healthy=True)
        log(f'santé après relance : {up}')
    finally:
        api.close()
    return down.get('status') != 'ok' and up.get('status') == 'ok'
