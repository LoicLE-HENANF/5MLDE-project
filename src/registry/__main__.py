"""Ligne de commande du registre.

uv run python -m registry statut volatilite
uv run python -m registry retour-arriere volatilite
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from registry.promotion import RegistryError, rollback, status


def main(argv: Sequence[str] | None = None) -> int:
    from connectors.config import MlflowConfig, PostgresConfig
    from connectors.mlflow import MlflowConnector
    from connectors.postgres import PostgresConnector

    parser = argparse.ArgumentParser(prog='registry', description='Alias des modèles enregistrés.')
    parser.add_argument('action', choices=['statut', 'retour-arriere'])
    parser.add_argument('modele', help='nom du modèle enregistré (ex. volatilite)')
    args = parser.parse_args(argv)

    connector = MlflowConnector(MlflowConfig.from_env())
    if args.action == 'retour-arriere':
        try:
            with PostgresConnector(PostgresConfig.from_env()) as db:
                version = rollback(connector, args.modele, record=db.record_model_event)
        except RegistryError as exc:
            print(exc)
            return 1
        print(f'{args.modele} : champion remis en version {version.version}')
    for alias, version in status(connector, args.modele).items():
        print(f'  {alias:<11} {version or "-"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
