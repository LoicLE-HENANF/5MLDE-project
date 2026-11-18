"""Relevé `docker stats` (CPU, mémoire) des conteneurs pendant une mesure de charge."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

_UNITS = {'B': 1 / 2**20, 'KiB': 1 / 2**10, 'MiB': 1.0, 'GiB': 2**10, 'kB': 1e3 / 2**20, 'MB': 1e6 / 2**20}
_SIZE = re.compile(r'^\s*([\d.]+)\s*([A-Za-z]+)')


def _mebibytes(usage: str) -> float | None:
    """'123.4MiB / 7.6GiB' -> 123.4 (mémoire utilisée, en Mio)."""
    match = _SIZE.match(usage.split('/')[0])
    if not match or match.group(2) not in _UNITS:
        return None
    return float(match.group(1)) * _UNITS[match.group(2)]


def parse_docker_stats(text: str) -> dict[str, dict[str, float | None]]:
    """CPU moyen / max (100 % = un cœur) et mémoire max par conteneur, depuis `docker stats` en JSON."""
    samples: dict[str, list[tuple[float, float | None]]] = {}
    for line in re.findall(r'\{[^{}]*\}', text):
        row = json.loads(line)
        try:
            cpu = float(row['CPUPerc'].rstrip('%'))
        except (KeyError, ValueError):
            continue
        samples.setdefault(row['Name'], []).append((cpu, _mebibytes(row.get('MemUsage', ''))))
    summary = {}
    for name, rows in samples.items():
        cpus = [cpu for cpu, _ in rows]
        mems = [mem for _, mem in rows if mem is not None]
        summary[name] = {
            'cpu_mean': sum(cpus) / len(cpus),
            'cpu_max': max(cpus),
            'mem_max': max(mems) if mems else None,
        }
    return summary


@contextmanager
def docker_stats(containers: Sequence[str]) -> Iterator[dict[str, dict[str, float | None]]]:
    """Relève `docker stats` en continu pendant le bloc ; le résumé est rempli à la sortie."""
    summary: dict[str, dict[str, float | None]] = {}
    with tempfile.TemporaryFile() as out:
        process = subprocess.Popen(
            ['docker', 'stats', '--format', '{{json .}}', *containers],
            stdout=out,
            stderr=subprocess.DEVNULL,
        )
        try:
            yield summary
        finally:
            process.kill()
            process.wait()
            out.seek(0)
            summary.update(parse_docker_stats(out.read().decode('utf-8', errors='replace')))
