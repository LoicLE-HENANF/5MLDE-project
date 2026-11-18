"""Connecteur MLflow.

- Unitaires : MlflowClient simulé + transport HTTP simulé (aucun serveur).
- Bout en bout local : vrai MLflow sur une base SQLite temporaire (même code serveur que la
  stack, sans réseau) pour valider run -> log_model -> registre -> alias -> chargement.
"""

from __future__ import annotations

import time
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import mlflow
import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST

from connectors import mlflow as mlflow_connector
from connectors.config import MlflowConfig
from connectors.errors import ConfigError, ConnectorError
from connectors.mlflow import CODE_ARCHIVE_ARTIFACT, REQUIREMENTS_ARTIFACT, MlflowConnector
from lineage import provenance
from lineage.errors import ProvenanceError
from tests.conftest import refuse


@pytest.fixture
def config(env) -> MlflowConfig:
    return MlflowConfig.from_env(env)


@pytest.fixture
def client() -> MagicMock:
    return MagicMock(spec=mlflow.MlflowClient)


# -- santé -------------------------------------------------------------------------------------
def test_check_ok(config, client, http_client):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, text='OK')

    MlflowConnector(config, client=client, http=http_client(handler)).check()
    assert seen == ['http://mlflow.test:5000/health']


def test_check_server_down(config, client, http_client):
    with pytest.raises(ConnectorError, match=r'MLflow \(http://mlflow.test:5000/health\) : injoignable'):
        MlflowConnector(config, client=client, http=http_client(refuse)).check()


def test_check_wrong_service_on_port(config, client, http_client):
    connector = MlflowConnector(config, client=client, http=http_client(lambda r: httpx.Response(404)))
    with pytest.raises(ConnectorError, match='HTTP 404'):
        connector.check()


def test_check_timeout(config, client, http_client):
    def handler(request):
        raise httpx.ReadTimeout('lent', request=request)

    with pytest.raises(ConnectorError, match='délai dépassé'):
        MlflowConnector(config, client=client, http=http_client(handler)).check(timeout=0.1)


def test_check_separate_registry(env, client, http_client):
    env['MLFLOW_REGISTRY_URI'] = 'http://registry.test:5000'
    seen = []

    def handler(request):
        seen.append(request.url.host)
        return httpx.Response(200)

    MlflowConnector(MlflowConfig.from_env(env), client=client, http=http_client(handler)).check()
    assert seen == ['mlflow.test', 'registry.test']


def test_constructor_does_no_io(config):
    MlflowConnector(config)  # client réel, serveur inexistant : aucune erreur à la construction


# -- expérience --------------------------------------------------------------------------------
def test_experiment_created_once(config, client):
    client.get_experiment_by_name.return_value = None
    client.create_experiment.return_value = '7'
    connector = MlflowConnector(config, client=client)
    assert connector.experiment_id() == '7'
    assert connector.experiment_id() == '7'
    client.create_experiment.assert_called_once_with('exp-test', tags={'projet': '5mlde-project'})


def test_existing_experiment_reused(config, client):
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id='3', lifecycle_stage='active')
    assert MlflowConnector(config, client=client).experiment_id() == '3'
    client.create_experiment.assert_not_called()


def test_deleted_experiment_is_config_error(config, client):
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id='3', lifecycle_stage='deleted')
    with pytest.raises(ConfigError, match='supprimée'):
        MlflowConnector(config, client=client).experiment_id()


def test_run_url(config, client):
    url = MlflowConnector(config, client=client).run_url('abc', experiment_id='3')
    assert url == 'http://mlflow.test:5000/#/experiments/3/runs/abc'


def test_log_model_outside_run_is_refused(config, client):
    with pytest.raises(ProvenanceError, match='start_run'):
        MlflowConnector(config, client=client).log_model(mlflow.pyfunc.log_model)


# -- registre ----------------------------------------------------------------------------------
def test_get_version_missing_returns_none(config, client):
    client.get_model_version_by_alias.side_effect = MlflowException(
        'absent', error_code=RESOURCE_DOES_NOT_EXIST
    )
    assert MlflowConnector(config, client=client).get_version('m', 'champion') is None


def test_get_version_other_errors_propagate(config, client):
    client.get_model_version_by_alias.side_effect = MlflowException('boom', error_code=INTERNAL_ERROR)
    with pytest.raises(MlflowException):
        MlflowConnector(config, client=client).get_version('m', 'champion')


def test_register_without_provenance_is_refused(config, client):
    client.get_logged_model.return_value = SimpleNamespace(tags={})
    info = SimpleNamespace(model_id='m-1', model_uri='models:/m-1')
    with pytest.raises(ProvenanceError, match='provenance absente'):
        MlflowConnector(config, client=client).register_model(info, 'rendement')


# -- bout en bout sur un vrai MLflow local -----------------------------------------------------
class _Constant(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return [42] * len(model_input)


RELEASED = provenance.CodeProvenance(
    version='0.2.0',
    git_ref='v0.2.0',
    dirty=False,
    editable=False,
    code_dirs=provenance.package_dirs(),
    code_sha256='a' * 64,
    requirements=('mlflow==3.16.1',),
    requirements_sha256='b' * 64,
)


@pytest.fixture
def local_connector(tmp_path, monkeypatch):
    """Vrai MLflow (SQLite temporaire) : même logique serveur que la stack, sans réseau."""
    monkeypatch.chdir(tmp_path)  # artefacts locaux dans tmp_path
    uri = f'sqlite:///{(tmp_path / "mlflow.db").as_posix()}'
    config = MlflowConfig(tracking_uri=uri, registry_uri=uri, experiment='e2e')
    yield MlflowConnector(config)
    mlflow.set_tracking_uri(None)
    mlflow.set_registry_uri(None)


@pytest.fixture
def released(monkeypatch) -> provenance.CodeProvenance:
    """Code « publié » (tag exact, arbre propre, installation non éditable)."""
    monkeypatch.setattr(provenance, 'current', lambda: RELEASED)
    return RELEASED


def test_end_to_end_run_registry_alias_load(local_connector, released, monkeypatch):
    monkeypatch.setattr(mlflow_connector, 'SYSTEM_METRICS_INTERVAL', 0.2)
    with local_connector.start_run('train', tags={'prefect.flow_run_id': 'fr-1', 'code.version': 'x'}) as run:
        mlflow.log_metric('rmse', 1.5)
        time.sleep(1.0)  # laisse passer au moins un échantillon de métriques système
        info = local_connector.log_model(mlflow.pyfunc.log_model, python_model=_Constant())

    stored = local_connector.client.get_run(run.info.run_id)
    assert stored.info.status == 'FINISHED'
    assert stored.data.tags['prefect.flow_run_id'] == 'fr-1'
    assert stored.data.metrics['rmse'] == 1.5
    for key, value in released.tags().items():  # la provenance l'emporte sur les tags fournis
        assert stored.data.tags[key] == value
    requirements = mlflow.artifacts.load_text(f'runs:/{run.info.run_id}/{REQUIREMENTS_ARTIFACT}')
    assert requirements.split() == ['mlflow==3.16.1']
    # l'archive du code exécuté est jointe au run
    archive = Path(mlflow.artifacts.download_artifacts(f'runs:/{run.info.run_id}/{CODE_ARCHIVE_ARTIFACT}'))
    with zipfile.ZipFile(archive) as code_zip:
        assert 'lineage/provenance.py' in code_zip.namelist()
    # métriques système enregistrées (au moins un échantillon à la fermeture du run)
    assert any(key.startswith('system/') for key in stored.data.metrics)

    # le modèle embarque le code du paquet et les dépendances exactes
    model_dir = Path(mlflow.artifacts.download_artifacts(info.model_uri))
    assert (model_dir / 'code' / 'lineage' / 'provenance.py').is_file()
    assert (model_dir / 'requirements.txt').read_text().split() == ['mlflow==3.16.1']

    assert local_connector.get_version('rendement', 'champion') is None  # modèle inexistant
    version = local_connector.register_model(info, 'rendement', alias='champion')
    assert str(version.version) == '1'
    assert version.tags['code.git_ref'] == 'v0.2.0'
    assert local_connector.get_version('rendement', 'challenger') is None  # alias inexistant
    assert local_connector.get_version('rendement', 'champion').run_id == run.info.run_id

    model = local_connector.load_model('rendement', 'champion')
    assert list(model.predict([[0], [1]])) == [42, 42]


@pytest.mark.parametrize('change', [{'dirty': True}, {'editable': True}])
def test_unreleased_code_cannot_be_registered(local_connector, monkeypatch, change):
    monkeypatch.setattr(provenance, 'current', lambda: replace(RELEASED, **change))
    with local_connector.start_run('train'):
        info = local_connector.log_model(mlflow.pyfunc.log_model, python_model=_Constant())
    with pytest.raises(ProvenanceError, match='non enregistré'):
        local_connector.register_model(info, 'rendement')
    version = local_connector.register_model(info, 'rendement', allow_unreleased=True)
    assert version.tags[f'code.{next(iter(change))}'] == 'true'


def test_log_model_rejects_managed_arguments(local_connector, released):
    with local_connector.start_run('train'), pytest.raises(ValueError, match='pip_requirements'):
        local_connector.log_model(mlflow.pyfunc.log_model, python_model=_Constant(), pip_requirements=[])


def test_end_to_end_failed_block_marks_run_failed(local_connector, released):
    with pytest.raises(RuntimeError), local_connector.start_run('train') as run:
        raise RuntimeError('entraînement cassé')
    assert local_connector.client.get_run(run.info.run_id).info.status == 'FAILED'
