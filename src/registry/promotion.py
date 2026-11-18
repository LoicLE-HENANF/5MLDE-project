"""Promotion et retour arrière des modèles dans le registre MLflow.

Alias utilisés pour chaque modèle enregistré :
- `champion` : version servie par l'API ;
- `challenger` : version qui tourne en parallèle sans être servie (comparaison en conditions réelles) ;
- `precedent` : ancien champion, cible d'un retour arrière.

Règles de décision (`decide`) pour une nouvelle version :
1. niveau 1 (statistique) échoué -> rien n'est enregistré ;
2. niveau 2 (stratégie) échoué -> enregistrée comme `challenger` ;
3. le champion actuel, rescoré sur le même test final, fait aussi bien ou mieux -> `challenger` ;
4. sinon -> `champion`, l'ancien champion devient `precedent`.

Une version enregistrée peut aussi être promue à la main (`promote`), avec une raison
obligatoire : la règle automatique est contournée mais la décision reste tracée.

Chaque enregistrement, promotion et retrait est tracé dans `tracing.model_events`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

CHAMPION = 'champion'
CHALLENGER = 'challenger'
PREVIOUS = 'precedent'

EventRecorder = Callable[..., Any]
"""Signature de `PostgresConnector.record_model_event` (arguments nommés)."""


class RegistryError(RuntimeError):
    """Opération de registre impossible (ex. retour arrière sans version précédente)."""


@dataclass(frozen=True)
class Decision:
    alias: str | None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def registered(self) -> bool:
        return self.alias is not None

    def summary(self) -> str:
        outcome = self.alias or 'rejeté'
        return f'{outcome} : ' + ' ; '.join(self.reasons)


def decide(*, level1: bool, level2: bool, score: float, champion_score: float | None) -> Decision:
    """Décide de l'alias d'une nouvelle version (score : plus bas = meilleur, même test final)."""
    if not level1:
        return Decision(None, ('niveau 1 échoué : ne bat pas la meilleure référence naïve',))
    reasons = ['niveau 1 validé']
    if not level2:
        reasons.append('niveau 2 échoué : la stratégie ne bat pas celles des références')
        return Decision(CHALLENGER, tuple(reasons))
    reasons.append('niveau 2 validé')
    if champion_score is not None and not score < champion_score:
        reasons.append(f'ne bat pas le champion actuel ({score:.4f} >= {champion_score:.4f})')
        return Decision(CHALLENGER, tuple(reasons))
    if champion_score is None:
        reasons.append('aucun champion actuel')
    else:
        reasons.append(f'bat le champion actuel ({score:.4f} < {champion_score:.4f})')
    return Decision(CHAMPION, tuple(reasons))


def model_uri(name: str, version: int | str) -> str:
    return f'models:/{name}/{version}'


def publish(
    connector,
    info,
    name: str,
    decision: Decision,
    *,
    record: EventRecorder | None = None,
    allow_unreleased: bool = False,
):
    """Enregistre la version selon la décision et déplace les alias ; renvoie la version ou None."""
    if not decision.registered:
        return None
    version = connector.register_model(
        info, name, tags={'registre.decision': decision.summary()}, allow_unreleased=allow_unreleased
    )
    run_id = version.run_id
    _record(
        record,
        name,
        version.version,
        'registered',
        run_id=run_id,
        details={'raisons': list(decision.reasons)},
    )
    if decision.alias == CHAMPION:
        previous = connector.get_version(name, CHAMPION)
        if previous is not None:
            connector.set_alias(name, PREVIOUS, previous.version)
            _record(record, name, previous.version, 'retired', alias=CHAMPION, run_id=previous.run_id)
    connector.set_alias(name, decision.alias, version.version)
    _record(record, name, version.version, 'promoted', alias=decision.alias, run_id=run_id)
    return version


def promote(connector, name: str, version: int | str, *, reason: str, record: EventRecorder | None = None):
    """Promotion manuelle d'une version enregistrée en `champion` ; renvoie la version promue.

    L'ancien champion devient `precedent` ; si la version promue était `challenger`, l'alias est retiré.
    """
    if not reason.strip():
        raise RegistryError('promotion manuelle : une raison est obligatoire')
    try:
        target = connector.client.get_model_version(name, str(version))
    except Exception as exc:
        raise RegistryError(f'{name} : version {version} introuvable') from exc
    current = connector.get_version(name, CHAMPION)
    if current is not None and str(current.version) == str(target.version):
        raise RegistryError(f'{name} : la version {target.version} est déjà {CHAMPION!r}')
    challenger = connector.get_version(name, CHALLENGER)
    if current is not None:
        connector.set_alias(name, PREVIOUS, current.version)
        _record(
            record,
            name,
            current.version,
            'retired',
            alias=CHAMPION,
            run_id=current.run_id,
            details={'manuel': True},
        )
    connector.set_alias(name, CHAMPION, target.version)
    if challenger is not None and str(challenger.version) == str(target.version):
        connector.client.delete_registered_model_alias(name, CHALLENGER)
    _record(
        record,
        name,
        target.version,
        'promoted',
        alias=CHAMPION,
        run_id=target.run_id,
        details={'manuel': True, 'raison': reason.strip()},
    )
    return target


def rollback(connector, name: str, *, record: EventRecorder | None = None):
    """Remet l'ancien champion (`precedent`) en service ; renvoie la version restaurée."""
    previous = connector.get_version(name, PREVIOUS)
    if previous is None:
        raise RegistryError(f'{name} : aucune version {PREVIOUS!r} vers laquelle revenir')
    current = connector.get_version(name, CHAMPION)
    connector.set_alias(name, CHAMPION, previous.version)
    connector.client.delete_registered_model_alias(name, PREVIOUS)
    if current is not None:
        _record(
            record,
            name,
            current.version,
            'retired',
            alias=CHAMPION,
            run_id=current.run_id,
            details={'retour_arriere': True},
        )
    _record(
        record,
        name,
        previous.version,
        'promoted',
        alias=CHAMPION,
        run_id=previous.run_id,
        details={'retour_arriere': True},
    )
    return previous


def status(connector, name: str) -> dict[str, str | None]:
    """Version pointée par chaque alias (None si absent)."""
    result = {}
    for alias in (CHAMPION, CHALLENGER, PREVIOUS):
        version = connector.get_version(name, alias)
        result[alias] = None if version is None else str(version.version)
    return result


def _record(
    record: EventRecorder | None,
    name: str,
    version: int | str,
    event: str,
    *,
    alias: str | None = None,
    run_id: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    if record is None:
        return
    record(
        model_name=name,
        version=version,
        event=event,
        model_uri=model_uri(name, version),
        alias=alias,
        mlflow_run_id=run_id,
        details=dict(details or {}),
    )
