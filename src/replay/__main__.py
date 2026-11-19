"""Rejeu de N mois d'exploitation : `uv run python -m replay --start 2025-11 --months 12`.

Le registre utilisé est `<modèle>-replay` et toutes les traces portent `mode = 'replay'` :
le champion live et ses tableaux de bord ne sont jamais touchés. `--reset` efface d'abord
les traces du rejeu (compte administrateur PostgreSQL, api_writer n'ayant pas DELETE).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace

from connectors.config import MlflowConfig, PostgresConfig, load_env
from connectors.mlflow import MlflowConnector
from connectors.postgres import PostgresConnector
from replay.models import SPECS, spec
from replay.runner import Replay
from replay.store import ReplayStore, reset


def _admin(config: PostgresConfig, env) -> PostgresConfig:
    return replace(config, user=env.get('POSTGRES_USER', 'postgres'), password=env['POSTGRES_PASSWORD'])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='replay', description="Rejoue des mois d'exploitation simulée.")
    parser.add_argument('--start', required=True, help='premier mois rejoué (AAAA-MM)')
    parser.add_argument('--months', type=int, default=12, help='nombre de mois consécutifs (défaut : 12)')
    parser.add_argument(
        '--modele', choices=sorted(SPECS), action='append', help='modèle(s) rejoué(s), défaut : tous'
    )
    parser.add_argument('--reset', action='store_true', help='efface les traces replay avant de commencer')
    parser.add_argument(
        '--non-publie', action='store_true', help='autorise un code non publié (essai uniquement)'
    )
    args = parser.parse_args(argv)

    env = load_env()
    config = PostgresConfig.from_env(env)
    if args.reset:
        with PostgresConnector(_admin(config, env)) as admin:
            reset(admin)
        print('traces du rejeu effacées (les registres MLflow *-replay sont à supprimer à la main)')

    from training.volatility import load_candles

    candles = load_candles(env)
    connector = MlflowConnector(MlflowConfig.from_env(env))
    with PostgresConnector(config) as db:
        store = ReplayStore(db)
        for name in args.modele or sorted(SPECS):
            replay = Replay(spec(name), connector, store, candles, allow_unreleased=args.non_publie)
            replay.run(args.start, args.months)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
