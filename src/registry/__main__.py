"""Ligne de commande du registre.

uv run python -m registry statut volatilite
uv run python -m registry retour-arriere volatilite
uv run python -m registry promouvoir volatilite --version 2 --raison "..."
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from registry.promotion import RegistryError, promote, rollback, status


def main(argv: Sequence[str] | None = None) -> int:
    from connectors.config import MlflowConfig, PostgresConfig
    from connectors.mlflow import MlflowConnector
    from connectors.postgres import PostgresConnector

    parser = argparse.ArgumentParser(prog='registry', description='Alias des modèles enregistrés.')
    parser.add_argument('action', choices=['statut', 'retour-arriere', 'promouvoir'])
    parser.add_argument('modele', help='nom du modèle enregistré (ex. volatilite)')
    parser.add_argument('--version', help='version à promouvoir (promouvoir)')
    parser.add_argument('--raison', default='', help='justification tracée (promouvoir, obligatoire)')
    args = parser.parse_args(argv)
    if args.action == 'promouvoir' and (not args.version or not args.raison.strip()):
        parser.error('promouvoir exige --version et --raison')

    connector = MlflowConnector(MlflowConfig.from_env())
    if args.action == 'retour-arriere':
        try:
            with PostgresConnector(PostgresConfig.from_env()) as db:
                version = rollback(connector, args.modele, record=db.record_model_event)
        except RegistryError as exc:
            print(exc)
            return 1
        print(f'{args.modele} : champion remis en version {version.version}')
    if args.action == 'promouvoir':
        try:
            with PostgresConnector(PostgresConfig.from_env()) as db:
                version = promote(
                    connector, args.modele, args.version, reason=args.raison, record=db.record_model_event
                )
        except RegistryError as exc:
            print(exc)
            return 1
        print(f'{args.modele} : version {version.version} promue champion à la main')
    for alias, version in status(connector, args.modele).items():
        print(f'  {alias:<11} {version or "-"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
