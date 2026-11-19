"""Boucle du rejeu : un entraînement par mois, puis prévisions, retours, santé et retour arrière.

Chaque mois simulé enchaîne exactement les étapes du live, avec le code de production :

1. entraînement à `as_of` = début du mois (seules les bougies clôturées avant comptent) ;
2. règle de promotion du rejeu (`replay_decision`) dans le registre `<modèle>-replay`,
   amorçage manuel tracé si aucun champion n'existe encore ;
3. jour par jour : prévisions des versions `champion` et `challenger`, retours des horizons clôturés,
   santé du jour J-2 (même délai que le live) et retour arrière après des jours dégradés consécutifs.

Un mois terminé est marqué dans `tracing.pipeline_runs` : une relance reprend au premier mois manquant.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import mlflow
import pandas as pd

from features import FEATURE_COLUMNS
from flows.feedback import ROLLBACK_AFTER_DAYS, should_roll_back
from registry.promotion import (
    CHALLENGER,
    CHAMPION,
    Decision,
    RegistryError,
    decide,
    model_uri,
    promote,
    publish,
    rollback,
)
from replay import batch
from replay.models import ModelSpec
from replay.store import ReplayStore
from serving.feedback import pending_feedback

HEALTH_DELAY = timedelta(days=2)
FEEDBACK_LOOKBACK = timedelta(days=30)
SCORE_WINDOW = timedelta(days=7)
BOOTSTRAP_REASON = 'rejeu : amorçage (aucun champion, premier modèle enregistré)'
LEVEL2_FAILED = 'niveau 2 échoué (non bloquant dans le rejeu) : la stratégie ne bat pas celles des références'


def replay_decision(*, level1: bool, level2: bool, score: float, champion_score: float | None) -> Decision:
    """Règle du rejeu (démonstration du cycle de vie) : niveau 1 + bat le champion du moment.

    Le niveau 2 est calculé et tracé dans les raisons mais ne bloque pas ; le live garde la règle complète.
    """
    decision = decide(level1=level1, level2=True, score=score, champion_score=champion_score)
    if not decision.registered or level2:
        return decision
    reasons = [LEVEL2_FAILED if reason == 'niveau 2 validé' else reason for reason in decision.reasons]
    return Decision(decision.alias, tuple(reasons))


@dataclass
class MonthReport:
    month: str
    decision: str
    predictions: int = 0
    feedback: int = 0
    health: int = 0
    rollbacks: list[int] = field(default_factory=list)

    def details(self) -> dict[str, Any]:
        return {
            'decision': self.decision,
            'predictions': self.predictions,
            'retours': self.feedback,
            'sante': self.health,
            'retours_arriere': self.rollbacks,
        }


class Replay:
    """Rejoue `spec` mois par mois ; `connector` est un `MlflowConnector`, `store` un `ReplayStore`."""

    def __init__(
        self,
        spec: ModelSpec,
        connector: Any,
        store: ReplayStore,
        candles: pd.DataFrame,
        *,
        allow_unreleased: bool = False,
        log: Callable[[str], None] = print,
    ) -> None:
        self.spec = spec
        self.connector = connector
        self.store = store
        self.candles = candles
        self.allow_unreleased = allow_unreleased
        self.log = log
        self._models: dict[int, Any] = {}

    # -- registre ---------------------------------------------------------------------------------
    def _model(self, version: int) -> Any:
        if version not in self._models:
            self.connector.activate()
            self._models[version] = mlflow.pyfunc.load_model(model_uri(self.spec.replay_name, version))
        return self._models[version]

    def _champion_score(self, result: Any) -> float | None:
        champion = self.connector.get_version(self.spec.replay_name, CHAMPION)
        if champion is None:
            return None
        holdout = self.spec.holdout(result)
        predicted = self._model(int(champion.version)).predict(holdout.loc[:, list(FEATURE_COLUMNS)])
        return self.spec.champion_score(pd.Series(predicted).to_numpy(dtype='float64'), holdout)

    def train(self, as_of: pd.Timestamp) -> str:
        result = self.spec.fit(batch.closed_before(self.candles, as_of), as_of)
        _, info = self.spec.log(result, self.connector)
        decision = replay_decision(
            level1=self.spec.level1(result),
            level2=self.spec.level2(result),
            score=self.spec.score(result),
            champion_score=self._champion_score(result),
        )
        version = publish(
            self.connector,
            info,
            self.spec.replay_name,
            decision,
            record=self.store.record_event,
            allow_unreleased=self.allow_unreleased,
        )
        summary = decision.summary()
        if version is not None and self.connector.get_version(self.spec.replay_name, CHAMPION) is None:
            promote(
                self.connector,
                self.spec.replay_name,
                version.version,
                reason=BOOTSTRAP_REASON,
                record=self.store.record_event,
            )
            summary += ' ; amorçage manuel en champion'
        self.log(f'{self.spec.replay_name} {as_of:%Y-%m} : {summary}')
        return summary

    # -- un jour ----------------------------------------------------------------------------------
    def _serving(self) -> list[tuple[str, Any]]:
        serving = []
        for alias in (CHAMPION, CHALLENGER):
            version = self.connector.get_version(self.spec.replay_name, alias)
            if version is not None:
                serving.append((alias, version))
        return serving

    def _predict_day(self, features: pd.DataFrame, cache: dict[int, pd.Series], start, end) -> int:
        written = 0
        for alias, version in self._serving():
            number = int(version.version)
            if number not in cache:
                cache[number] = batch.predict_rows(self._model(number).predict, features)
            rows = self.spec.rows(features, cache[number], start, end, with_positions=alias == CHAMPION)
            written += self.store.write_predictions(
                self.spec.name, number, alias, self.spec.horizon_hours, version.run_id, rows
            )
        return written

    def _feedback(self, now: pd.Timestamp) -> int:
        waiting = self.store.pending(
            self.spec.name, (now - FEEDBACK_LOOKBACK).to_pydatetime(), now.to_pydatetime()
        )
        due = pending_feedback(waiting, now)
        if due.empty:
            return 0
        scored = self.spec.feedback(due, batch.closed_before(self.candles, now), as_of=now)
        return self.store.write_feedback(scored)

    def _health(self, day: date) -> int:
        end = pd.Timestamp(datetime.combine(day, datetime.min.time(), tzinfo=UTC) + timedelta(days=1))
        scored = self.store.scored(self.spec.name, (end - SCORE_WINDOW).to_pydatetime(), end.to_pydatetime())
        if scored.empty:
            return 0
        return self.store.write_health(self.spec.health(scored, self.candles, day=day))

    def _rollback(self, day: date) -> int | None:
        champion = self.connector.get_version(self.spec.replay_name, CHAMPION)
        if champion is None:
            return None
        history = self.store.health_history(
            self.spec.name, int(champion.version), self.spec.metric, day, ROLLBACK_AFTER_DAYS
        )
        if not should_roll_back(history):
            return None
        try:
            restored = rollback(self.connector, self.spec.replay_name, record=self.store.record_event)
        except RegistryError as exc:
            self.log(f'{self.spec.replay_name} {day} : retour arrière impossible ({exc})')
            return None
        self.log(
            f'{self.spec.replay_name} {day} : champion v{champion.version} dégradé, '
            f'retour à la v{restored.version}'
        )
        return int(restored.version)

    # -- un mois ----------------------------------------------------------------------------------
    def month(self, start: pd.Timestamp, end: pd.Timestamp) -> MonthReport:
        report = MonthReport(month=f'{start:%Y-%m}', decision=self.train(start))
        features = batch.month_features(self.candles, start, end)
        cache: dict[int, pd.Series] = {}
        day = start
        while day < end:
            following = min(day + pd.Timedelta(days=1), end)
            report.predictions += self._predict_day(features, cache, day, following)
            report.feedback += self._feedback(following)
            health_day = (following - HEALTH_DELAY).date()
            if health_day >= start.date():
                report.health += self._health(health_day)
                restored = self._rollback(health_day)
                if restored is not None:
                    report.rollbacks.append(restored)
            day = following
        self.log(
            f'{self.spec.replay_name} {report.month} : {report.predictions} prévisions, '
            f'{report.feedback} retours, {report.health} lignes de santé, '
            f'retours arrière {report.rollbacks or "aucun"}'
        )
        return report

    def run(self, first: str, months: int) -> list[MonthReport]:
        starts = batch.month_starts(first, months)
        done = self.store.done_months(self.spec.name)
        reports = []
        ends = [*starts[1:], starts[-1] + pd.DateOffset(months=1)]
        for start, end in zip(starts, ends, strict=True):
            month = f'{start:%Y-%m}'
            if month in done:
                self.log(f'{self.spec.replay_name} {month} : déjà rejoué, ignoré')
                continue
            began = datetime.now(UTC)
            report = self.month(start, pd.Timestamp(end))
            self.store.mark_month(self.spec.name, month, began, datetime.now(UTC), report.details())
            reports.append(report)
        return reports
