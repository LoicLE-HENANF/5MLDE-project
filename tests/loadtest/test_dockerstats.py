"""Lecture de `docker stats` (sortie JSON en continu, avec codes d'effacement d'écran)."""

from __future__ import annotations

import json

import pytest

from loadtest.dockerstats import parse_docker_stats


def _line(name, cpu, mem):
    return json.dumps({'Name': name, 'CPUPerc': cpu, 'MemUsage': mem})


def test_parse_averages_cpu_and_keeps_peak_memory():
    text = '\x1b[2J\x1b[H'.join(
        [
            _line('api', '50.00%', '100MiB / 7.6GiB'),
            _line('pg', '5.00%', '1.5GiB / 7.6GiB'),
            _line('api', '150.00%', '120.5MiB / 7.6GiB'),
            _line('api', '--', '-- / --'),  # conteneur en démarrage : ignoré
        ]
    )
    summary = parse_docker_stats(text)
    assert summary['api'] == {'cpu_mean': pytest.approx(100.0), 'cpu_max': 150.0, 'mem_max': 120.5}
    assert summary['pg']['mem_max'] == pytest.approx(1536.0)


def test_parse_handles_unknown_units_and_empty_output():
    assert parse_docker_stats(_line('api', '1%', '3PiB / 4PiB'))['api']['mem_max'] is None
    assert parse_docker_stats('') == {}
