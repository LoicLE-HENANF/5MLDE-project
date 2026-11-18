"""Intégration : dashboards et alertes provisionnés, requêtes exécutées par Grafana (grafana_reader).

Chaque requête SQL des fichiers versionnés passe par l'API de Grafana avec des valeurs de
variables plausibles : une colonne renommée ou un droit manquant fait échouer le test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration

INFRA = Path(__file__).resolve().parents[2] / 'infra' / 'grafana'
VARIABLES = {'$modele': 'volatilite', '$symbole': "'BTCUSDT'"}


def _queries() -> list[tuple[str, str, str, str]]:
    """(dashboard, titre, datasource, sql) pour chaque panneau, variable et annotation."""
    queries = []
    for path in sorted((INFRA / 'dashboards').glob('*.json')):
        board = json.loads(path.read_text(encoding='utf-8'))
        for panel in board['panels']:
            for query in panel['targets']:
                queries.append((board['uid'], panel['title'], query['datasource']['uid'], query['rawSql']))
        for variable in board['templating']['list']:
            queries.append(
                (
                    board['uid'],
                    f'variable {variable["name"]}',
                    variable['datasource']['uid'],
                    variable['query'],
                )
            )
        for annotation in board['annotations']['list']:
            if 'target' in annotation:
                queries.append(
                    (
                        board['uid'],
                        f'annotation {annotation["name"]}',
                        annotation['datasource']['uid'],
                        annotation['target']['rawSql'],
                    )
                )
    alerts = yaml.safe_load((INFRA / 'provisioning' / 'alerting' / 'alertes.yml').read_text(encoding='utf-8'))
    for group in alerts['groups']:
        for rule in group['rules']:
            for query in rule['data']:
                if query['datasourceUid'] != '__expr__':
                    queries.append(
                        ('alerte', rule['title'], query['datasourceUid'], query['model']['rawSql'])
                    )
    return queries


def test_dashboards_are_provisioned(grafana):
    found = {
        d['uid']: d['folderTitle']
        for d in grafana._request('GET', '/api/search', params={'tag': '5mlde'}).json()
    }
    assert {'projet-traitements', 'projet-modeles', 'projet-service'} <= set(found)
    assert set(found.values()) == {'Projet 5MLDE'}


def test_alert_rules_are_provisioned_and_healthy(grafana):
    expected = {
        rule['uid']
        for group in yaml.safe_load(
            (INFRA / 'provisioning' / 'alerting' / 'alertes.yml').read_text(encoding='utf-8')
        )['groups']
        for rule in group['rules']
    }
    loaded = {rule['uid'] for rule in grafana._request('GET', '/api/v1/provisioning/alert-rules').json()}
    assert expected <= loaded
    groups = grafana._request('GET', '/api/prometheus/grafana/api/v1/rules').json()['data']['groups']
    errors = {r['name']: r.get('lastError') for g in groups for r in g['rules'] if r.get('health') == 'error'}
    assert not errors


@pytest.mark.parametrize(
    'board, title, datasource, sql',
    _queries(),
    ids=lambda v: v if isinstance(v, str) and len(v) < 40 else None,
)
def test_query_runs_through_grafana(grafana, board, title, datasource, sql):
    for name, value in VARIABLES.items():
        sql = sql.replace(name, value)
    response = grafana._request(
        'POST',
        '/api/ds/query',
        accept=(400, 500),
        json={
            'from': 'now-30d',
            'to': 'now',
            'queries': [{'refId': 'A', 'datasource': {'uid': datasource}, 'rawSql': sql, 'format': 'table'}],
        },
    )
    result = response.json().get('results', {}).get('A', {})
    assert response.status_code == 200 and not result.get('error'), (
        f'{board} / {title} : {result.get("error") or response.text[:300]}'
    )
