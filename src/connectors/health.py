"""Vérifie la connexion à chaque brique de la stack.

    uv run python -m connectors.health                 # toutes les briques
    uv run python -m connectors.health mlflow postgres # seulement celles-ci

Code de sortie 0 si tout est joignable, 1 sinon (utilisable en CI ou au démarrage d'un service).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping

from connectors.config import GrafanaConfig, MlflowConfig, PostgresConfig, PrefectConfig, load_env
from connectors.errors import ConfigError, ConnectorError
from connectors.grafana import GrafanaConnector
from connectors.mlflow import MlflowConnector
from connectors.postgres import PostgresConnector
from connectors.prefect import PrefectConnector


def _mlflow(env: Mapping[str, str]) -> None:
    MlflowConnector(MlflowConfig.from_env(env)).check()


def _prefect(env: Mapping[str, str]) -> None:
    PrefectConnector(PrefectConfig.from_env(env)).check()


def _postgres(env: Mapping[str, str]) -> None:
    PostgresConnector(PostgresConfig.from_env(env)).check()


def _grafana(env: Mapping[str, str]) -> None:
    with GrafanaConnector(GrafanaConfig.from_env(env)) as grafana:
        grafana.check()


CHECKS: dict[str, Callable[[Mapping[str, str]], None]] = {
    'mlflow': _mlflow,
    'prefect': _prefect,
    'postgres': _postgres,
    'grafana': _grafana,
}


def run_checks(names: list[str], env: Mapping[str, str]) -> dict[str, str | None]:
    """Nom de la brique -> None si OK, sinon le message d'erreur."""
    results: dict[str, str | None] = {}
    for name in names:
        try:
            CHECKS[name](env)
            results[name] = None
        except (ConfigError, ConnectorError) as exc:
            results[name] = str(exc)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        'services', nargs='*', metavar='SERVICE', help=f'parmi {", ".join(CHECKS)} (défaut : tous)'
    )
    args = parser.parse_args(argv)
    unknown = sorted(set(args.services) - set(CHECKS))
    if unknown:
        parser.error(f'service(s) inconnu(s) : {", ".join(unknown)}')
    results = run_checks(args.services or list(CHECKS), load_env())
    for name, error in results.items():
        print(f'[{"OK" if error is None else "KO"}] {name}' + ('' if error is None else f'\n     {error}'))
    return 0 if all(error is None for error in results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
