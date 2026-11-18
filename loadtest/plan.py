"""Plan du test de charge : configurations, scénarios, SLO et lecture des résultats.

Module pur (sans Locust ni Docker) : c'est la partie testée. Trois profils d'utilisateurs :
les bots de trading lisent `GET /predictions/latest` (en rafale au début de chaque heure, quand
la nouvelle prévision tombe), les analystes consultent l'API et le registre, les opérateurs
surveillent `/health` et `/system/state`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

# Cible à tenir (specs du projet, à valider) : 200 bots au pic, 20 analystes, 2 opérateurs.
TARGET_BOTS = 200
ANALYSTS = 20
OPERATORS = 2
MARGIN = 2.0  # la configuration retenue doit encore tenir le SLO avec MARGIN x TARGET_BOTS bots


@dataclass(frozen=True)
class Slo:
    """Objectif de service : p95 sous le seuil et taux d'erreur sous le plafond."""

    p95_ms: float = 200.0
    max_error_rate: float = 0.01


@dataclass(frozen=True)
class Config:
    """Configuration de l'API : processus uvicorn et limite CPU du conteneur (en cœurs)."""

    workers: int
    cpus: float

    def __post_init__(self) -> None:
        if self.workers <= 0 or self.cpus <= 0:
            raise ValueError('workers et cpus doivent être strictement positifs')

    @property
    def label(self) -> str:
        return f'{self.workers}w / {self.cpus:g} CPU'

    @property
    def cost(self) -> tuple[float, int]:
        """Ordre de coût : d'abord le CPU réservé, puis le nombre de processus (mémoire)."""
        return self.cpus, self.workers


CONFIGS = tuple(Config(workers, cpus) for cpus in (0.5, 1.0) for workers in (1, 2, 4))


@dataclass(frozen=True)
class Step:
    """Un palier de charge : nombre d'utilisateurs par profil, montée et durée de mesure.

    `measure_spawn` : la mesure inclut la montée (cas du pic, où la rafale est l'objet du test) ;
    sinon elle commence une fois tous les utilisateurs lancés.
    """

    bots: int
    analysts: int
    operators: int
    spawn_rate: float
    duration_s: float
    measure_spawn: bool = False

    def __post_init__(self) -> None:
        if min(self.bots, self.analysts, self.operators) < 0 or self.users == 0:
            raise ValueError('effectifs négatifs ou palier vide')
        if self.spawn_rate <= 0 or self.duration_s <= 0:
            raise ValueError('spawn_rate et duration_s doivent être strictement positifs')

    @property
    def users(self) -> int:
        return self.bots + self.analysts + self.operators


# Régime normal : quelques bots qui interrogent l'API entre deux bougies.
NORMAL = Step(bots=20, analysts=ANALYSTS, operators=OPERATORS, spawn_rate=10, duration_s=60)
# Pic de début d'heure : tous les bots arrivent en 10 s, mesure de la rafale comprise.
PEAK = Step(
    bots=TARGET_BOTS, analysts=ANALYSTS, operators=OPERATORS, spawn_rate=22, duration_s=60, measure_spawn=True
)
RAMP_START, RAMP_STEP, RAMP_MAX = 100, 100, 1500
RAMP_DURATION_S = 20


def ramp(start: int = RAMP_START, step: int = RAMP_STEP, maximum: int = RAMP_MAX) -> Iterator[Step]:
    """Montée progressive jusqu'à rupture : paliers de bots croissants (s'arrête au premier échec)."""
    if start <= 0 or step <= 0 or maximum < start:
        raise ValueError('rampe invalide')
    for bots in range(start, maximum + 1, step):
        yield Step(
            bots=bots, analysts=ANALYSTS, operators=OPERATORS, spawn_rate=100, duration_s=RAMP_DURATION_S
        )


@dataclass(frozen=True)
class Measure:
    """Mesure Locust d'un palier (toutes routes confondues, `endpoints` en détail)."""

    step: Step
    duration_s: float
    requests: int
    failures: int
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    max_ms: float | None
    endpoints: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def rps(self) -> float:
        return self.requests / self.duration_s

    @property
    def error_rate(self) -> float:
        return self.failures / self.requests if self.requests else 1.0

    def meets(self, slo: Slo) -> bool:
        """SLO tenu : des requêtes ont abouti, p95 et taux d'erreur sous les seuils."""
        return (
            self.requests > 0
            and self.p95_ms is not None
            and self.p95_ms < slo.p95_ms
            and self.error_rate < slo.max_error_rate
        )


def breaking_point(measures: Sequence[Measure], slo: Slo) -> Measure | None:
    """Dernier palier tenant le SLO avant le premier échec (None si le premier palier échoue)."""
    last = None
    for measure in measures:
        if not measure.meets(slo):
            break
        last = measure
    return last


@dataclass(frozen=True)
class ConfigResult:
    """Résultats d'une configuration : pic et rampe (le régime normal est informatif)."""

    config: Config
    peak: Measure
    ramp: tuple[Measure, ...]

    def capacity(self, slo: Slo) -> int:
        """Bots tenus par la rampe en respectant le SLO (0 si le premier palier échoue)."""
        point = breaking_point(self.ramp, slo)
        return 0 if point is None else point.step.bots


@dataclass(frozen=True)
class Recommendation:
    config: Config | None
    reason: str


def recommend(
    results: Iterable[ConfigResult],
    slo: Slo,
    *,
    target_bots: int = TARGET_BOTS,
    margin: float = MARGIN,
) -> Recommendation:
    """Plus petite configuration qui tient le pic et `margin` x la cible sur la rampe."""
    results = sorted(results, key=lambda r: r.config.cost)
    if not results:
        raise ValueError('aucun résultat')
    needed = margin * target_bots
    for result in results:
        if result.peak.meets(slo) and result.capacity(slo) >= needed:
            return Recommendation(
                result.config,
                f'plus petite configuration qui tient le pic ({target_bots} bots) et '
                f'{result.capacity(slo)} bots sur la rampe (≥ {needed:g}, marge {margin:g}x)',
            )
    best = max(results, key=lambda r: (r.capacity(slo), -r.config.cpus, -r.config.workers))
    return Recommendation(
        None,
        f'aucune configuration ne tient la marge {margin:g}x ; la plus robuste est {best.config.label} '
        f'({best.capacity(slo)} bots sur la rampe)',
    )
