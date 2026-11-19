"""Provenance du code : quelle version exacte du projet a produit un run ou un modèle.

Tout vient des métadonnées d'installation du paquet, sans appeler git à l'exécution :

- la version est calculée par hatch-vcs au build : `0.2.0` = tag `v0.2.0` ; `0.2.1.dev3+g1a2b3c4`
  = commit `1a2b3c4` ; suffixe `.d<date>` = arbre de travail modifié au moment du build ;
- une installation éditable (`uv sync` par défaut) exécute les sources telles qu'elles sont sur le
  disque, qui peuvent avoir changé depuis le calcul de la version : elle n'est pas publiable ;
- `uv run --no-editable ...` installe une copie figée du paquet : la version décrit exactement
  le code exécuté.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from lineage.errors import ProvenanceError

DISTRIBUTION = '5MLDE-project'
# Paquets importables du projet, copiés dans chaque modèle (tenu aligné sur
# [tool.hatch.build.targets.wheel].packages par tests/test_provenance.py).
PACKAGES = (
    'api',
    'app',
    'connectors',
    'features',
    'flows',
    'incident',
    'ingestion',
    'lineage',
    'marketdata',
    'registry',
    'replay',
    'serving',
    'strategies',
    'training',
)

TAG_PREFIX = 'code.'
REQUIRED_TAGS = ('version', 'git_ref', 'sha256', 'dirty', 'editable')


@dataclass(frozen=True)
class CodeProvenance:
    version: str
    git_ref: str  # 'v0.2.0' (tag) ou 'g1a2b3c4' (commit)
    dirty: bool
    editable: bool
    code_dirs: tuple[Path, ...]
    code_sha256: str
    requirements: tuple[str, ...]
    requirements_sha256: str

    def tags(self) -> dict[str, str]:
        """Tags MLflow décrivant ce code (posés sur les runs, modèles et versions)."""
        return {
            f'{TAG_PREFIX}version': self.version,
            f'{TAG_PREFIX}git_ref': self.git_ref,
            f'{TAG_PREFIX}dirty': str(self.dirty).lower(),
            f'{TAG_PREFIX}editable': str(self.editable).lower(),
            f'{TAG_PREFIX}sha256': self.code_sha256,
            f'{TAG_PREFIX}requirements_sha256': self.requirements_sha256,
        }


def parse_version(version: str) -> tuple[str, bool]:
    """(référence git, arbre modifié) d'une version hatch-vcs."""
    parsed = Version(version)
    if parsed.local is None:
        return f'v{parsed.public}', False
    parts = parsed.local.split('.')
    ref = next((p for p in parts if p.startswith('g')), None)
    dirty = any(p.startswith('d') and p[1:].isdigit() for p in parts)
    if ref is None:
        raise ProvenanceError(f'version {version!r} sans commit : paquet construit hors de git ?')
    return ref, dirty


def is_editable(dist_name: str = DISTRIBUTION) -> bool:
    raw = distribution(dist_name).read_text('direct_url.json')
    return bool(raw and json.loads(raw).get('dir_info', {}).get('editable'))


def package_dirs(packages: tuple[str, ...] = PACKAGES) -> tuple[Path, ...]:
    """Dossiers des paquets, résolus à côté de `lineage` (installés ensemble par la même roue).

    Pas de `find_spec` : un modèle MLflow chargé ajoute sa copie de code à `sys.path`,
    qui masquerait alors les sources réelles.
    """
    root = Path(__file__).resolve().parent.parent
    dirs = tuple(root / name for name in packages)
    for path in dirs:
        if not (path / '__init__.py').is_file():
            raise ProvenanceError(f'paquet {path.name!r} introuvable dans {root}')
    return dirs


def _tree_files(dirs: tuple[Path, ...]) -> Iterator[tuple[str, Path]]:
    """(chemin relatif, fichier) des paquets, triés, hors bytecode : base de l'empreinte et de l'archive."""
    for root in dirs:
        files = sorted(
            p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'
        )
        for path in files:
            yield f'{root.name}/{path.relative_to(root).as_posix()}', path


def hash_tree(dirs: tuple[Path, ...]) -> str:
    """SHA-256 du contenu des paquets (chemins relatifs + octets), hors bytecode."""
    digest = hashlib.sha256()
    for name, path in _tree_files(dirs):
        digest.update(f'{name}\0'.encode())
        digest.update(path.read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()


def archive_tree(dirs: tuple[Path, ...], destination: Path) -> Path:
    """Archive zip reproductible des paquets : exactement les fichiers couverts par `hash_tree`."""
    with zipfile.ZipFile(destination, 'w') as archive:
        for name, path in _tree_files(dirs):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return destination


def _platform_marker(requirement: Requirement) -> str | None:
    """Marqueur de plateforme à conserver (ceux qui portent sur les extras sont déjà résolus)."""
    if requirement.marker is None or 'extra' in str(requirement.marker):
        return None
    return str(requirement.marker)


def pinned_requirements(dist_name: str = DISTRIBUTION) -> tuple[str, ...]:
    """Dépendances d'exécution du projet, transitives, épinglées sur les versions installées."""
    root = canonicalize_name(dist_name)
    pins: dict[str, tuple[str, str | None]] = {}
    seen: set[tuple[str, frozenset[str]]] = set()
    stack: list[tuple[str, frozenset[str], str | None]] = [(dist_name, frozenset(), None)]
    while stack:
        name, extras, marker = stack.pop()
        key = canonicalize_name(name)
        try:
            dist = distribution(name)
        except PackageNotFoundError as exc:
            raise ProvenanceError(f'dépendance {name!r} requise mais non installée') from exc
        # Un seul chemin sans marqueur suffit pour que la dépendance soit inconditionnelle.
        if key != root and (key not in pins or marker is None):
            pins[key] = (f'{dist.metadata["Name"]}=={dist.version}', marker)
        if (key, extras) in seen:
            continue
        seen.add((key, extras))
        for raw in dist.requires or []:
            requirement = Requirement(raw)
            if requirement.marker is not None and not any(
                requirement.marker.evaluate({'extra': extra}) for extra in (*extras, '')
            ):
                continue
            stack.append((requirement.name, frozenset(requirement.extras), _platform_marker(requirement)))
    return tuple(
        sorted(
            (pin if marker is None else f'{pin}; {Marker(marker)}' for pin, marker in pins.values()),
            key=str.lower,
        )
    )


def current(dist_name: str = DISTRIBUTION, packages: tuple[str, ...] = PACKAGES) -> CodeProvenance:
    """Provenance du code en cours d'exécution."""
    try:
        version = distribution(dist_name).version
    except PackageNotFoundError as exc:
        raise ProvenanceError(f'{dist_name} non installé : lancer via `uv run`') from exc
    git_ref, dirty = parse_version(version)
    dirs = package_dirs(packages)
    requirements = pinned_requirements(dist_name)
    return CodeProvenance(
        version=version,
        git_ref=git_ref,
        dirty=dirty,
        editable=is_editable(dist_name),
        code_dirs=dirs,
        code_sha256=hash_tree(dirs),
        requirements=requirements,
        requirements_sha256=hashlib.sha256('\n'.join(requirements).encode()).hexdigest(),
    )


def release_problems(tags: dict[str, str]) -> list[str]:
    """Raisons pour lesquelles du code décrit par ces tags ne peut pas être publié ([] = publiable)."""
    missing = [TAG_PREFIX + k for k in REQUIRED_TAGS if TAG_PREFIX + k not in tags]
    if missing:
        return [
            f'provenance absente ({", ".join(missing)}) : logger le modèle avec '
            'MlflowConnector.log_model dans MlflowConnector.start_run'
        ]
    problems = []
    if tags[TAG_PREFIX + 'dirty'] != 'false':
        problems.append(f'code non commité (version {tags[TAG_PREFIX + "version"]}) : commiter puis relancer')
    if tags[TAG_PREFIX + 'editable'] != 'false':
        problems.append('installation éditable : relancer avec `uv run --no-editable`')
    return problems
