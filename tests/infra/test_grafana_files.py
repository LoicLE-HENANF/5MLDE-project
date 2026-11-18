"""Cohérence des fichiers Grafana versionnés : dashboards JSON et règles d'alerte provisionnées."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

INFRA = Path(__file__).resolve().parents[2] / 'infra' / 'grafana'
DASHBOARDS = sorted((INFRA / 'dashboards').glob('*.json'))
ALERTS = INFRA / 'provisioning' / 'alerting' / 'alertes.yml'


def _datasource_uids() -> set[str]:
    files = (INFRA / 'provisioning' / 'datasources').glob('*.yml')
    return {ds['uid'] for f in files for ds in yaml.safe_load(f.read_text(encoding='utf-8'))['datasources']}


def _dashboards() -> dict[str, dict]:
    boards = [json.loads(f.read_text(encoding='utf-8')) for f in DASHBOARDS]
    return {board['uid']: board for board in boards}


def _rules() -> list[dict]:
    return [
        rule
        for group in yaml.safe_load(ALERTS.read_text(encoding='utf-8'))['groups']
        for rule in group['rules']
    ]


def test_dashboards_are_present_with_unique_uids():
    boards = _dashboards()
    assert set(boards) == {'projet-traitements', 'projet-modeles', 'projet-service', 'projet-charge'}
    assert len(boards) == len(DASHBOARDS)


@pytest.mark.parametrize('path', DASHBOARDS, ids=lambda p: p.stem)
def test_panels_are_well_formed(path):
    board = json.loads(path.read_text(encoding='utf-8'))
    known = _datasource_uids()
    ids = [panel['id'] for panel in board['panels']]
    assert len(ids) == len(set(ids)), 'identifiants de panneau en double'
    for panel in board['panels']:
        assert panel['title'] and panel['targets'], panel
        assert panel['gridPos']['x'] >= 0 and panel['gridPos']['x'] + panel['gridPos']['w'] <= 24
        for query in panel['targets']:
            assert query['datasource']['uid'] in known, (panel['title'], query['datasource'])
            assert query['rawSql'].lstrip().upper().startswith('SELECT'), panel['title']


@pytest.mark.parametrize('path', DASHBOARDS, ids=lambda p: p.stem)
def test_template_variables_are_defined(path):
    board = json.loads(path.read_text(encoding='utf-8'))
    defined = {variable['name'] for variable in board['templating']['list']}
    sql = ' '.join(q['rawSql'] for p in board['panels'] for q in p['targets'])
    sql += ' '.join(a['target']['rawSql'] for a in board['annotations']['list'] if 'target' in a)
    used = set(re.findall(r'\$(?!__)(\w+)', sql))
    assert used <= defined, f'variables non définies : {used - defined}'


def test_alert_rules_point_to_existing_panels_and_datasources():
    boards = _dashboards()
    known = _datasource_uids() | {'__expr__'}
    rules = _rules()
    assert len({rule['uid'] for rule in rules}) == len(rules)
    for rule in rules:
        annotations = rule['annotations']
        board = boards[annotations['__dashboardUid__']]
        assert int(annotations['__panelId__']) in {panel['id'] for panel in board['panels']}, rule['uid']
        ref_ids = {query['refId'] for query in rule['data']}
        assert rule['condition'] in ref_ids, rule['uid']
        for query in rule['data']:
            assert query['datasourceUid'] in known, rule['uid']
            assert query['model']['refId'] == query['refId'], rule['uid']
