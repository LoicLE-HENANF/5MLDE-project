"""Connecteur Prefect.

- Unitaires : transport HTTP simulé, settings et run_deployment remplacés.
- Dans un vrai flow : serveur Prefect éphémère (SQLite temporaire), pour valider les
  helpers de contexte (ids de run, tags MLflow, artifacts) tels qu'un flow les verra.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest
from prefect import flow, task

from connectors import prefect as prefect_connector
from connectors.config import PrefectConfig
from connectors.errors import ConfigError, ConnectorError
from connectors.prefect import (
    PrefectConnector,
    get_logger,
    mlflow_tags,
    publish_markdown,
    run_context,
    task_options,
)
from tests.conftest import refuse


@pytest.fixture
def config(env) -> PrefectConfig:
    return PrefectConfig.from_env(env)


@pytest.fixture
def library_points_to_config(monkeypatch, config):
    monkeypatch.setattr(prefect_connector, '_settings_api_url', lambda: config.api_url)


# -- santé -------------------------------------------------------------------------------------
@pytest.mark.usefixtures('library_points_to_config')
def test_check_ok(config, http_client):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json=True)

    PrefectConnector(config, http=http_client(handler)).check()
    assert seen == ['http://prefect.test:4200/api/health']


@pytest.mark.usefixtures('library_points_to_config')
def test_check_server_down(config, http_client):
    with pytest.raises(ConnectorError, match=r'Prefect .* injoignable'):
        PrefectConnector(config, http=http_client(refuse)).check()


@pytest.mark.parametrize('library_url', [None, 'http://autre:4200/api'])
def test_check_detects_library_mismatch(config, http_client, monkeypatch, library_url):
    """La bibliothèque Prefect ne lit pas notre config : un écart = flows sur un autre serveur."""
    monkeypatch.setattr(prefect_connector, '_settings_api_url', lambda: library_url)
    connector = PrefectConnector(config, http=http_client(lambda r: httpx.Response(200)))
    with pytest.raises(ConfigError, match='Prefect utiliserait'):
        connector.check()


def test_reads_the_settings_the_library_uses():
    from prefect.settings import PREFECT_API_URL, temporary_settings

    with temporary_settings(updates={PREFECT_API_URL: 'http://lib:4200/api/'}):
        assert prefect_connector._settings_api_url() == 'http://lib:4200/api'


def test_flow_run_url(config):
    url = PrefectConnector(config).flow_run_url('1234')
    assert url == 'http://prefect.test:4200/runs/flow-run/1234'


def test_trigger_does_not_wait_by_default(config, monkeypatch):
    import prefect.deployments

    calls = {}
    monkeypatch.setattr(
        prefect.deployments, 'run_deployment', lambda **kwargs: calls.update(kwargs) or 'FLOW_RUN'
    )
    assert PrefectConnector(config).trigger('pipeline/prod', {'a': 1}) == 'FLOW_RUN'
    assert calls == {'name': 'pipeline/prod', 'parameters': {'a': 1}, 'timeout': 0, 'tags': None}


# -- hors d'un run -----------------------------------------------------------------------------
def test_helpers_outside_run():
    assert run_context() == {}
    assert mlflow_tags() == {}
    assert publish_markdown('# x', key='k') is None
    assert isinstance(get_logger(), logging.Logger)


def test_task_options_defaults_and_overrides():
    options = task_options(retries=2)
    assert options['retries'] == 2 and options['persist_result'] is False
    task(**options)(lambda: None)  # options acceptées par @task


# -- dans un vrai flow (serveur éphémère) ------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _wait_healthy(url: str, process: subprocess.Popen, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'serveur Prefect arrêté (code {process.returncode})')
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return
        except httpx.TransportError:  # ConnectError ET ConnectTimeout (boucle locale Windows)
            pass
        time.sleep(0.5)
    raise RuntimeError(f'serveur Prefect non joignable après {timeout:.0f} s : {url}')


@pytest.fixture(scope='module')
def prefect_server(tmp_path_factory):
    """Serveur Prefect éphémère (SQLite temporaire), équivalent de prefect_test_harness.

    prefect_test_harness ne rattrape que httpx.ConnectError pendant le démarrage ; sous Windows,
    un port de boucle locale pas encore ouvert produit un ConnectTimeout qui fait échouer le
    harness avant que le serveur ait fini de démarrer. On lance donc la même commande uvicorn
    et on attend la santé en tolérant toute erreur de transport.
    """
    import prefect
    from prefect.settings import PREFECT_API_URL, temporary_settings

    home = tmp_path_factory.mktemp('prefect-home')
    port = _free_port()
    env = {
        **os.environ,
        'PREFECT_HOME': str(home),
        'PREFECT_SERVER_DATABASE_CONNECTION_URL': f'sqlite+aiosqlite:///{(home / "prefect.db").as_posix()}',
        'PREFECT_SERVER_ANALYTICS_ENABLED': 'false',
        'PREFECT_UI_ENABLED': '0',
    }
    env.pop('PREFECT_API_URL', None)
    process = subprocess.Popen(
        [
            sys.executable,
            '-m',
            'uvicorn',
            '--app-dir',
            str(prefect.__module_path__.parent),
            '--factory',
            'prefect.server.api.server:create_app',
            '--host',
            '127.0.0.1',
            '--port',
            str(port),
            '--log-level',
            'error',
            '--lifespan',
            'on',
        ],
        env=env,
    )
    api_url = f'http://127.0.0.1:{port}/api'
    try:
        _wait_healthy(f'{api_url}/health', process)
        with temporary_settings(updates={PREFECT_API_URL: api_url}):
            yield api_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@pytest.mark.usefixtures('prefect_server')
def test_helpers_inside_flow_and_task(config):
    connector = PrefectConnector(config)

    @task(**task_options())
    def inner():
        return run_context()

    @flow(name='flow-test-connecteurs')
    def outer():
        get_logger().info('log du run')
        artifact_id = publish_markdown('# Rapport', key='rapport-test')
        return run_context(), inner(), mlflow_tags(connector), artifact_id

    flow_ids, task_ids, tags, artifact_id = outer()
    assert flow_ids['flow_name'] == 'flow-test-connecteurs'
    assert task_ids['flow_run_id'] == flow_ids['flow_run_id'] and 'task_run_id' in task_ids
    assert tags['prefect.flow_run_id'] == flow_ids['flow_run_id']
    assert tags['prefect.flow_run_url'].endswith(f'/runs/flow-run/{flow_ids["flow_run_id"]}')
    assert artifact_id is not None
