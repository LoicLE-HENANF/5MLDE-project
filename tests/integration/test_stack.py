"""Tests d'intégration contre la stack docker compose réelle (configuration lue dans .env).

    docker compose up -d --build --wait
    uv run pytest -m integration

Chaque test crée ses objets sous un suffixe unique et les supprime ensuite. Les traces
PostgreSQL sont effacées avec le compte administrateur (POSTGRES_PASSWORD) : api_writer n'a
volontairement pas le droit DELETE.
"""

from __future__ import annotations

import contextlib
import dataclasses
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import mlflow
import psycopg
import pytest
from prefect import flow
from prefect.client.orchestration import get_client
from prefect.settings import PREFECT_API_URL, temporary_settings

from connectors.config import GrafanaConfig, MlflowConfig, PostgresConfig, PrefectConfig, load_env
from connectors.grafana import GrafanaConnector
from connectors.mlflow import MlflowConnector
from connectors.postgres import PostgresConnector
from connectors.prefect import PrefectConnector, mlflow_tags, publish_markdown, run_context

pytestmark = pytest.mark.integration

SUFFIX = uuid.uuid4().hex[:8]
TEST_PIPELINE = f'integration-test-{SUFFIX}'
TEST_EXPERIMENT = 'integration-tests'
TEST_MODEL = f'integration-test-{SUFFIX}'


class _Constant(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return [42] * len(model_input)


# -- fixtures ----------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def env() -> dict[str, str]:
    return load_env()


@pytest.fixture(scope='module')
def mlflow_connector(env):
    config = dataclasses.replace(MlflowConfig.from_env(env), experiment=TEST_EXPERIMENT)
    connector = MlflowConnector(config)
    yield connector
    with contextlib.suppress(mlflow.exceptions.MlflowException):  # modèle absent si le test a échoué tôt
        connector.client.delete_registered_model(TEST_MODEL)


@pytest.fixture(scope='module')
def prefect_connector(env):
    connector = PrefectConnector(PrefectConfig.from_env(env))
    with temporary_settings(updates={PREFECT_API_URL: connector.config.api_url}):
        yield connector


@pytest.fixture(scope='module')
def db(env):
    connector = PostgresConnector(PostgresConfig.from_env(env))
    with connector:
        yield connector
    # nettoyage avec le superutilisateur : api_writer ne peut pas supprimer
    config = connector.config
    with psycopg.connect(
        host=config.host,
        port=config.port,
        dbname=config.dbname,
        user='postgres',
        password=env['POSTGRES_PASSWORD'],
        connect_timeout=5,
    ) as conn:
        conn.execute('DELETE FROM tracing.pipeline_runs WHERE pipeline = %s', (TEST_PIPELINE,))
        conn.execute('DELETE FROM tracing.model_events WHERE model_name = %s', (TEST_MODEL,))


@pytest.fixture(scope='module')
def grafana(env):
    with GrafanaConnector(GrafanaConfig.from_env(env)) as connector:
        yield connector


def grafana_sql(grafana: GrafanaConnector, datasource_uid: str, sql: str) -> list[list]:
    """Requête SQL exécutée PAR Grafana (rôle grafana_reader) ; renvoie les colonnes."""
    response = grafana._request(
        'POST',
        '/api/ds/query',
        json={
            'queries': [
                {
                    'refId': 'A',
                    'datasource': {'uid': datasource_uid},
                    'rawSql': sql,
                    'format': 'table',
                }
            ]
        },
    )
    frames = response.json()['results']['A']['frames']
    return frames[0]['data']['values'] if frames else []


# -- un service à la fois ----------------------------------------------------------------------
@pytest.mark.parametrize('name', ['mlflow', 'prefect', 'postgres', 'grafana'])
def test_health(name, mlflow_connector, prefect_connector, db, grafana):
    {'mlflow': mlflow_connector, 'prefect': prefect_connector, 'postgres': db, 'grafana': grafana}[
        name
    ].check()


def test_mlflow_run_registry_alias_load(mlflow_connector):
    """Backend PostgreSQL + artefacts servis par le serveur (--serve-artifacts)."""
    with mlflow_connector.start_run('integration', tags={'suffix': SUFFIX}) as run:
        mlflow.log_metric('rmse', 1.5)
        info = mlflow_connector.log_model(mlflow.pyfunc.log_model, python_model=_Constant())

    stored = mlflow_connector.client.get_run(run.info.run_id)
    assert stored.info.status == 'FINISHED' and stored.data.metrics['rmse'] == 1.5
    assert stored.info.artifact_uri.startswith('mlflow-artifacts:')  # pas de chemin local
    assert stored.data.tags['code.version']

    # les tests tournent sur l'installation éditable : enregistrement d'essai explicite
    version = mlflow_connector.register_model(info, TEST_MODEL, alias='champion', allow_unreleased=True)
    assert version.tags['code.sha256'] == stored.data.tags['code.sha256']
    assert mlflow_connector.get_version(TEST_MODEL, 'champion').version == version.version
    assert list(mlflow_connector.load_model(TEST_MODEL, 'champion').predict([[0], [1]])) == [42, 42]
    assert httpx.get(mlflow_connector.run_url(run.info.run_id), timeout=5).status_code == 200


def test_postgres_writes_visible_to_grafana(db, grafana):
    now = datetime.now(UTC)
    run_id = db.record_pipeline_run(
        pipeline=TEST_PIPELINE,
        status='success',
        started_at=now - timedelta(seconds=3),
        ended_at=now,
        details={'suffix': SUFFIX},
    )
    event_id = db.record_model_event(
        model_name=TEST_MODEL,
        version=1,
        event='promoted',
        alias='champion',
        model_uri=f'models:/{TEST_MODEL}/1',
    )

    runs = grafana_sql(
        grafana,
        'projet-traces',
        f"SELECT id, status FROM tracing.pipeline_runs WHERE pipeline = '{TEST_PIPELINE}'",
    )
    events = grafana_sql(
        grafana, 'projet-traces', f"SELECT id FROM tracing.model_events WHERE model_name = '{TEST_MODEL}'"
    )
    assert runs == [[run_id], ['success']]
    assert events == [[event_id]]


def test_grafana_read_only_datasources_reach_their_database(grafana):
    assert grafana_sql(grafana, 'projet-prefect', 'SELECT count(*) FROM flow')[0][0] >= 0
    assert grafana_sql(grafana, 'projet-mlflow', 'SELECT count(*) FROM experiments')[0][0] >= 1


def test_grafana_dashboard_and_annotation(grafana):
    uid = f'it-{SUFFIX}'
    url = grafana.upsert_dashboard(
        {'uid': uid, 'title': f'Integration {SUFFIX}', 'panels': []}, message='integration test'
    )
    try:
        assert url.endswith(f'/d/{uid}/integration-{SUFFIX}')
        annotation_id = grafana.annotate(
            f'integration {SUFFIX}', tags=['integration-test'], dashboard_uid=uid
        )
        grafana._request('DELETE', f'/api/annotations/{annotation_id}')
    finally:
        grafana._request('DELETE', f'/api/dashboards/uid/{uid}', accept=(404,))


# -- bout en bout : flow Prefect -> MLflow -> PostgreSQL -> Grafana ----------------------------
@flow(name=f'integration-test-{SUFFIX}', validate_parameters=False)  # paramètres = connecteurs
def integration_flow(connector: MlflowConnector, prefect: PrefectConnector, db: PostgresConnector) -> dict:
    started = datetime.now(UTC)
    with connector.start_run('integration-flow', tags=mlflow_tags(prefect)) as run:
        mlflow.log_metric('score', 0.5)
    ids = run_context()
    artifact_id = publish_markdown(
        f'[Run MLflow]({connector.run_url(run.info.run_id)})', key=f'integration-{SUFFIX}'
    )
    db.record_pipeline_run(
        pipeline=TEST_PIPELINE,
        status='success',
        started_at=started,
        ended_at=datetime.now(UTC),
        prefect_flow_run_id=ids['flow_run_id'],
        mlflow_run_id=run.info.run_id,
    )
    return {**ids, 'mlflow_run_id': run.info.run_id, 'artifact_id': str(artifact_id)}


def test_flow_links_prefect_mlflow_and_traces(mlflow_connector, prefect_connector, db, grafana):
    result = integration_flow(mlflow_connector, prefect_connector, db)
    try:
        tags = mlflow_connector.client.get_run(result['mlflow_run_id']).data.tags
        assert tags['prefect.flow_run_id'] == result['flow_run_id']
        assert result['artifact_id'] != 'None'

        flow_run = httpx.get(
            f'{prefect_connector.config.api_url}/flow_runs/{result["flow_run_id"]}', timeout=5
        ).json()
        assert flow_run['state']['type'] == 'COMPLETED'

        traced = grafana_sql(
            grafana,
            'projet-traces',
            'SELECT prefect_flow_run_id::text, mlflow_run_id FROM tracing.pipeline_runs '
            f"WHERE prefect_flow_run_id = '{result['flow_run_id']}'",
        )
        assert traced == [[result['flow_run_id']], [result['mlflow_run_id']]]
        # Grafana lit aussi les runs Prefect dans la base du serveur Prefect
        assert grafana_sql(
            grafana, 'projet-prefect', f"SELECT state_type FROM flow_run WHERE id = '{result['flow_run_id']}'"
        ) == [['COMPLETED']]
    finally:
        with get_client(sync_client=True) as client:
            client.delete_flow_run(result['flow_run_id'])


@flow(name=f'integration-trigger-{SUFFIX}')
def trigger_target(n: int = 1) -> int:
    return n


def test_trigger_creates_scheduled_run(prefect_connector):
    """trigger() via un vrai déploiement (aucun worker : le run reste planifié)."""
    deployment_id = trigger_target.to_deployment(name='integration').apply()
    try:
        run = prefect_connector.trigger(
            f'integration-trigger-{SUFFIX}/integration',
            parameters={'n': 2},
            wait_s=0,
            tags=['integration-test'],
        )
        assert run.state.type.value == 'SCHEDULED' and run.parameters == {'n': 2}
        assert prefect_connector.flow_run_url(run.id).endswith(str(run.id))
    finally:
        with get_client(sync_client=True) as client:
            client.delete_deployment(deployment_id)  # supprime aussi ses runs planifiés
