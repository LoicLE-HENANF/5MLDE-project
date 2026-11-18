"""Campagne de charge : chaque configuration de l'API passe les trois scénarios.

    uv run --group load python -m loadtest                 # campagne complète (~45 min)
    uv run --group load python -m loadtest --configs 1x1   # une configuration (workers x cpus)

Pour chaque configuration, l'API est redémarrée (`docker compose up -d --wait api` avec
API_WORKERS / API_CPUS), puis : régime normal, pic de début d'heure, rampe jusqu'à rupture.
Les résultats vont dans `tracing.load_tests` (dashboard Grafana « Charge ») et dans
`loadtest/results/<campagne>.json` ; l'API est remise dans sa configuration de départ à la fin.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from connectors.config import PostgresConfig, load_env
from connectors.postgres import PostgresConnector
from loadtest.bench import run_step, wait_ready, write_json
from loadtest.dockerstats import docker_stats
from loadtest.plan import CONFIGS, NORMAL, PEAK, Config, ConfigResult, Measure, Slo, Step, ramp, recommend

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'loadtest' / 'results'
API, POSTGRES = '5mlde-project-api-1', '5mlde-project-postgres-1'


def _parse_config(text: str) -> Config:
    workers, _, cpus = text.partition('x')
    return Config(int(workers), float(cpus))


def deploy(config: Config) -> None:
    """Redémarre l'API avec la configuration donnée et attend qu'elle soit saine."""
    env = {**os.environ, 'API_WORKERS': str(config.workers), 'API_CPUS': f'{config.cpus:g}'}
    subprocess.run(
        ['docker', 'compose', 'up', '-d', '--wait', '--no-deps', 'api'], cwd=ROOT, env=env, check=True
    )


def _measure(step: Step, host: str) -> tuple[Measure, dict[str, Any]]:
    with docker_stats([API, POSTGRES]) as stats:
        measure, extra = run_step(step, host)
    return measure, {**extra, 'docker': dict(stats)}


def _row(
    campaign: str,
    scenario: str,
    config: Config,
    measure: Measure,
    extra: dict[str, Any],
    slo: Slo,
    *,
    users: int | None = None,
) -> dict[str, Any]:
    api, pg = extra['docker'].get(API, {}), extra['docker'].get(POSTGRES, {})
    return {
        'campaign': campaign,
        'scenario': scenario,
        'workers': config.workers,
        'cpus': config.cpus,
        'users': measure.step.users if users is None else users,
        'duration_s': measure.duration_s,
        'requests': measure.requests,
        'failures': measure.failures,
        'p50_ms': measure.p50_ms,
        'p95_ms': measure.p95_ms,
        'p99_ms': measure.p99_ms,
        'max_ms': measure.max_ms,
        'api_cpu_pct': api.get('cpu_mean'),
        'api_mem_mb': api.get('mem_max'),
        'pg_cpu_pct': pg.get('cpu_mean'),
        'pg_mem_mb': pg.get('mem_max'),
        'slo_ok': measure.meets(slo),
        'details': {
            'bots': measure.step.bots,
            'endpoints': measure.endpoints,
            'api_cpu_max_pct': api.get('cpu_max'),
            'spawn_s': extra['spawn_s'],
            'client_cpu_warning': extra['client_cpu_warning'],
            'errors': extra['errors'],
        },
    }


def _print(label: str, measure: Measure, slo: Slo) -> None:
    p95 = '-' if measure.p95_ms is None else f'{measure.p95_ms:.0f} ms'
    verdict = 'ok' if measure.meets(slo) else 'HORS SLO'
    print(
        f'  {label:<18} {measure.step.users:>5} util. {measure.rps:>7.1f} req/s  p95 {p95:>8}  '
        f'erreurs {100 * measure.error_rate:5.2f} %  {verdict}',
        flush=True,
    )


def run_config(campaign: str, config: Config, host: str, slo: Slo) -> tuple[ConfigResult, list[dict]]:
    deploy(config)
    wait_ready(host)
    print(f'{config.label}', flush=True)
    rows = []
    normal, extra = _measure(NORMAL, host)
    _print('normal', normal, slo)
    rows.append(_row(campaign, 'normal', config, normal, extra, slo))
    peak, extra = _measure(PEAK, host)
    _print('pic', peak, slo)
    rows.append(_row(campaign, 'pic', config, peak, extra, slo))
    steps, ramp_extra = [], []
    for step in ramp():
        measure, extra = _measure(step, host)
        _print(f'rampe {step.bots} bots', measure, slo)
        steps.append(measure)
        ramp_extra.append(extra)
        if not measure.meets(slo):
            break
    result = ConfigResult(config, peak, tuple(steps))
    # La ligne « rupture » décrit le dernier palier tenu (ou le premier, en échec, si aucun ne tient).
    held = [m for m in steps if m.meets(slo)]
    index = len(held) - 1 if held else 0
    row = _row(campaign, 'rupture', config, steps[index], ramp_extra[index], slo, users=result.capacity(slo))
    row['details']['ramp'] = [
        {
            'bots': m.step.bots,
            'rps': m.rps,
            'p95_ms': m.p95_ms,
            'error_rate': m.error_rate,
            'slo_ok': m.meets(slo),
            'api_cpu_pct': e['docker'].get(API, {}).get('cpu_mean'),
        }
        for m, e in zip(steps, ramp_extra, strict=True)
    ]
    rows.append(row)
    return result, rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='loadtest', description="Test de charge et dimensionnement de l'API."
    )
    parser.add_argument('--host', default='http://127.0.0.1:8000', help="adresse de l'API")
    parser.add_argument(
        '--configs', nargs='*', type=_parse_config, help='configurations workersxcpus (ex. 2x1)'
    )
    parser.add_argument('--campaign', default=datetime.now(UTC).strftime('%Y-%m-%dT%H%MZ'))
    parser.add_argument('--no-record', action='store_true', help='ne pas écrire dans tracing.load_tests')
    args = parser.parse_args(argv)
    slo = Slo()
    env = load_env()
    initial = Config(int(env.get('API_WORKERS', '1')), float(env.get('API_CPUS', '1')))
    results, rows = [], []
    try:
        for config in args.configs or CONFIGS:
            result, config_rows = run_config(args.campaign, config, args.host, slo)
            results.append(result)
            rows.extend(config_rows)
    finally:
        print(f'remise en place de la configuration de départ ({initial.label})', flush=True)
        deploy(initial)
    choice = recommend(results, slo)
    for row in rows:
        row['recommended'] = choice.config is not None and (row['workers'], row['cpus']) == (
            choice.config.workers,
            choice.config.cpus,
        )
    print(f'\nrecommandation : {choice.config.label if choice.config else "aucune"} — {choice.reason}')
    write_json(
        RESULTS / f'{args.campaign.replace(":", "")}.json',
        {'campaign': args.campaign, 'slo': asdict(slo), 'recommendation': choice.reason, 'rows': rows},
    )
    if not args.no_record:
        with PostgresConnector(PostgresConfig.from_env(env)) as db:
            for row in rows:
                db.record_load_test(**row)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
