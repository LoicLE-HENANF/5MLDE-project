"""Incident contrôlé : `uv run python -m incident lot-invalide` ou `... dependance`.

`lot-invalide` envoie un lot corrompu dans le vrai flow d'ingestion (rejeté, tracé, alertes).
`dependance` arrête MLflow, montre l'API dégradée, puis le relance (Entrée) et vérifie le retour.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from incident.scenarios import dependance, lot_invalide


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='incident', description='Joue un incident contrôlé.')
    sub = parser.add_subparsers(dest='scenario', required=True)
    sub.add_parser('lot-invalide', help="lot de bougies corrompu dans le flow d'ingestion")
    dep = sub.add_parser('dependance', help="arrêt puis relance d'un service (défaut : mlflow)")
    dep.add_argument('--service', default='mlflow', help='service docker compose (défaut : mlflow)')
    args = parser.parse_args(argv)
    ok = lot_invalide() if args.scenario == 'lot-invalide' else dependance(args.service)
    print('incident joué comme attendu' if ok else 'résultat inattendu, voir ci-dessus')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
