"""Provenance du code : version hatch-vcs, empreintes, dépendances épinglées, règles de publication."""

from __future__ import annotations

import tomllib
from importlib.metadata import version as installed_version
from pathlib import Path

import pytest

from connectors.config import ROOT
from lineage import provenance
from lineage.errors import ProvenanceError


@pytest.mark.parametrize(
    ('version', 'expected'),
    [
        ('0.2.0', ('v0.2.0', False)),
        ('0.2.1.dev3+g1a2b3c4', ('g1a2b3c4', False)),
        ('0.2.1.dev3+g1a2b3c4.d20260106', ('g1a2b3c4', True)),
    ],
)
def test_parse_version(version, expected):
    assert provenance.parse_version(version) == expected


def test_parse_version_without_commit_is_refused():
    with pytest.raises(ProvenanceError, match='sans commit'):
        provenance.parse_version('0.1.dev0+d20260106')


def _tree(root: Path, files: dict[str, str]) -> Path:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
    return root


def test_hash_tree_is_deterministic_and_ignores_bytecode(tmp_path):
    a = _tree(tmp_path / 'a' / 'pkg', {'x.py': '1', 'sub/y.py': '2'})
    b = _tree(tmp_path / 'b' / 'pkg', {'sub/y.py': '2', 'x.py': '1', '__pycache__/x.cpython-312.pyc': '?'})
    assert provenance.hash_tree((a,)) == provenance.hash_tree((b,))


@pytest.mark.parametrize('files', [{'x.py': '1 ', 'sub/y.py': '2'}, {'x.py': '1', 'sub/z.py': '2'}])
def test_hash_tree_detects_content_and_name_changes(tmp_path, files):
    reference = _tree(tmp_path / 'ref' / 'pkg', {'x.py': '1', 'sub/y.py': '2'})
    changed = _tree(tmp_path / 'new' / 'pkg', files)
    assert provenance.hash_tree((reference,)) != provenance.hash_tree((changed,))


def test_package_dirs_point_to_sources():
    assert provenance.package_dirs() == (
        ROOT / 'src' / 'connectors',
        ROOT / 'src' / 'lineage',
        ROOT / 'src' / 'marketdata',
    )


def test_packages_match_wheel_configuration():
    pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    wheel = pyproject['tool']['hatch']['build']['targets']['wheel']['packages']
    assert [Path(p).name for p in wheel] == list(provenance.PACKAGES)


def test_pinned_requirements_are_exact_and_transitive():
    pins = provenance.pinned_requirements()
    names = {pin.split('==')[0].split(';')[0].lower() for pin in pins}
    assert f'mlflow=={installed_version("mlflow")}' in pins
    assert {'mlflow', 'prefect', 'httpx', 'sqlalchemy', 'pydantic'} <= names  # directes et transitives
    assert all('==' in pin for pin in pins)
    assert 'pytest' not in names and 'ruff' not in names  # groupe dev exclu
    assert list(pins) == sorted(pins, key=str.lower)


def test_current_describes_installed_code():
    code = provenance.current()
    assert code.version == installed_version(provenance.DISTRIBUTION)
    assert code.git_ref == provenance.parse_version(code.version)[0]
    assert len(code.code_sha256) == len(code.requirements_sha256) == 64
    assert set(code.tags()) == {provenance.TAG_PREFIX + k for k in provenance.REQUIRED_TAGS} | {
        provenance.TAG_PREFIX + 'requirements_sha256'
    }


def test_current_without_installed_distribution():
    with pytest.raises(ProvenanceError, match='non installé'):
        provenance.current(dist_name='paquet-absent')


def _tags(**overrides: str) -> dict[str, str]:
    base = {'version': '0.2.0', 'git_ref': 'v0.2.0', 'sha256': 'a', 'dirty': 'false', 'editable': 'false'}
    return {provenance.TAG_PREFIX + k: v for k, v in {**base, **overrides}.items()}


def test_release_problems():
    assert provenance.release_problems(_tags()) == []
    assert 'non commité' in provenance.release_problems(_tags(dirty='true'))[0]
    assert '--no-editable' in provenance.release_problems(_tags(editable='true'))[0]
    assert len(provenance.release_problems(_tags(dirty='true', editable='true'))) == 2
    assert 'provenance absente' in provenance.release_problems({})[0]
