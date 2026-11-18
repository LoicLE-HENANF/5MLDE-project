"""Exécution d'un palier Locust (en bibliothèque) et relevé `docker stats` pendant la mesure."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import gevent
from locust.env import Environment
from locust.stats import StatsEntry

from loadtest.plan import Measure, Step
from loadtest.users import Analyst, Bot, Operator, sized


def _ms(entry: StatsEntry, percentile: float) -> float | None:
    return float(entry.get_response_time_percentile(percentile)) if entry.num_requests else None


def _endpoints(env: Environment) -> dict[str, dict[str, Any]]:
    return {
        name: {
            'requests': entry.num_requests,
            'failures': entry.num_failures,
            'p50_ms': _ms(entry, 0.5),
            'p95_ms': _ms(entry, 0.95),
            'p99_ms': _ms(entry, 0.99),
        }
        for (name, _method), entry in sorted(env.stats.entries.items())
    }


def run_step(step: Step, host: str) -> tuple[Measure, dict[str, Any]]:
    """Lance le palier contre l'API et renvoie la mesure Locust (et des informations annexes)."""
    classes = [
        sized(base, count)
        for base, count in ((Bot, step.bots), (Analyst, step.analysts), (Operator, step.operators))
        if count
    ]
    env = Environment(user_classes=classes, host=host)
    runner = env.create_local_runner()
    runner.start(step.users, spawn_rate=step.spawn_rate)
    spawn_started = time.monotonic()
    if not step.measure_spawn:
        runner.spawning_greenlet.join()
        env.stats.reset_all()
    started = time.monotonic()
    gevent.sleep(step.duration_s)
    elapsed = time.monotonic() - started
    total = env.stats.total
    measure = Measure(
        step=step,
        duration_s=elapsed,
        requests=total.num_requests,
        failures=total.num_failures,
        p50_ms=_ms(total, 0.5),
        p95_ms=_ms(total, 0.95),
        p99_ms=_ms(total, 0.99),
        max_ms=float(total.max_response_time) if total.num_requests else None,
        endpoints=_endpoints(env),
    )
    errors = sorted({str(error.error)[:200] for error in env.stats.errors.values()})[:5]
    extra = {
        'spawn_s': round(started - spawn_started, 1),
        # Locust et l'API partagent la machine : un client saturé fausse la mesure.
        'client_cpu_warning': bool(getattr(runner, 'cpu_warning_emitted', False)),
        'errors': errors,
    }
    runner.quit()
    return measure, extra


def wait_ready(host: str, timeout_s: float = 120.0) -> None:
    """Attend que l'API réponde 200 sur /health (après un redémarrage), puis la chauffe."""
    import requests  # dépendance de Locust

    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if requests.get(f'{host}/health', timeout=5).status_code == 200:
                break
        except requests.RequestException:
            pass
        if time.monotonic() > deadline:
            raise RuntimeError(f"l'API ne répond pas sur {host}/health")
        gevent.sleep(2)
    for _ in range(20):
        requests.get(f'{host}/predictions/latest', timeout=10)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')
