"""Worker Prefect : sert les déploiements planifiés du projet dans un seul processus.

Tourne dans le service `worker` de compose.yml (image du projet, installation non éditable).
Les plannings sont en UTC et décalés pour que chaque flow trouve les données du précédent :

| Déploiement                          | Cron       | Rôle                                            |
|--------------------------------------|------------|-------------------------------------------------|
| ingest-candles/horaire               | 2 * * * *  | bougies de l'heure écoulée, archives absentes   |
| predict-volatility/horaire-prevision | 7 * * * *  | prévision 24 h du champion (et du challenger)   |
| feedback-volatility/horaire-retours  | 20 * * * * | retours à horizon clôturé, santé, retour arrière |
| train-volatility/hebdomadaire        | 30 3 * * 1 | réentraînement du lundi et règle de promotion   |

Un déploiement peut aussi être lancé à la main depuis l'UI Prefect (« Quick run »).

    uv run --no-editable python -m flows.serve
"""

from __future__ import annotations

from prefect import serve
from prefect.schedules import Cron

from flows.feedback import feedback_volatility
from flows.ingest import ingest_candles
from flows.predict import predict_volatility
from flows.train import train_volatility

TIMEZONE = 'UTC'
SCHEDULES = {
    'horaire': (ingest_candles, '2 * * * *'),
    'horaire-prevision': (predict_volatility, '7 * * * *'),
    'horaire-retours': (feedback_volatility, '20 * * * *'),
    'hebdomadaire': (train_volatility, '30 3 * * 1'),
}
TAGS = ['5mlde-project']


def deployments() -> list:
    return [
        flow.to_deployment(name=name, schedule=Cron(cron, timezone=TIMEZONE), tags=TAGS)
        for name, (flow, cron) in SCHEDULES.items()
    ]


def main() -> int:
    # Un run à la fois par déploiement suffit : les flows sont idempotents et rapides.
    serve(*deployments(), limit=4)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
