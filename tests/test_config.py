from __future__ import annotations

import pytest

from connectors.config import GrafanaConfig, MlflowConfig, PostgresConfig, PrefectConfig, load_env
from connectors.errors import ConfigError


class TestMlflowConfig:
    def test_valid(self, env):
        config = MlflowConfig.from_env(env)
        assert config.tracking_uri == 'http://mlflow.test:5000'
        assert config.registry_uri == config.tracking_uri
        assert config.experiment == 'exp-test'

    def test_trailing_slash_removed(self, env):
        env['MLFLOW_TRACKING_URI'] = 'http://mlflow.test:5000/'
        assert MlflowConfig.from_env(env).tracking_uri == 'http://mlflow.test:5000'

    def test_missing_uri_has_no_silent_default(self, env):
        del env['MLFLOW_TRACKING_URI']
        with pytest.raises(ConfigError, match='MLFLOW_TRACKING_URI : variable absente'):
            MlflowConfig.from_env(env)

    @pytest.mark.parametrize(
        'value', ['./mlruns', 'sqlite:///mlflow.db', 'file:///tmp/mlruns', 'localhost:5000', 'http://']
    )
    def test_non_http_uri_rejected(self, env, value):
        env['MLFLOW_TRACKING_URI'] = value
        with pytest.raises(ConfigError, match='URL http'):
            MlflowConfig.from_env(env)

    def test_separate_registry(self, env):
        env['MLFLOW_REGISTRY_URI'] = 'http://registry.test:5000'
        assert MlflowConfig.from_env(env).registry_uri == 'http://registry.test:5000'


class TestPrefectConfig:
    def test_valid_and_ui_derived(self, env):
        config = PrefectConfig.from_env(env)
        assert config.api_url == 'http://prefect.test:4200/api'
        assert config.ui_url == 'http://prefect.test:4200'

    def test_api_suffix_required(self, env):
        env['PREFECT_API_URL'] = 'http://prefect.test:4200'
        with pytest.raises(ConfigError, match='/api'):
            PrefectConfig.from_env(env)

    def test_missing(self, env):
        del env['PREFECT_API_URL']
        with pytest.raises(ConfigError, match='PREFECT_API_URL'):
            PrefectConfig.from_env(env)


class TestPostgresConfig:
    def test_valid_with_defaults(self, env):
        config = PostgresConfig.from_env(env)
        assert (config.host, config.port, config.dbname, config.user) == (
            'db.test',
            5432,
            'mlops',
            'api_writer',
        )
        assert config.target == 'api_writer@db.test:5432/mlops'

    def test_password_not_in_repr(self, env):
        assert 'secret-api' not in repr(PostgresConfig.from_env(env))

    def test_missing_password(self, env):
        del env['API_DB_PASSWORD']
        with pytest.raises(ConfigError, match='API_DB_PASSWORD'):
            PostgresConfig.from_env(env)

    def test_all_errors_reported_at_once(self, env):
        env.update(POSTGRES_PORT='abc', API_DB_POOL_MAX='0')
        del env['API_DB_PASSWORD']
        with pytest.raises(ConfigError) as info:
            PostgresConfig.from_env(env)
        message = str(info.value)
        assert 'POSTGRES_PORT' in message and 'API_DB_POOL_MAX' in message and 'API_DB_PASSWORD' in message

    @pytest.mark.parametrize('port', ['0', '70000'])
    def test_port_out_of_range(self, env, port):
        env['POSTGRES_PORT'] = port
        with pytest.raises(ConfigError, match='POSTGRES_PORT'):
            PostgresConfig.from_env(env)

    def test_pool_max_below_min(self, env):
        env.update(API_DB_POOL_MIN='4', API_DB_POOL_MAX='2')
        with pytest.raises(ConfigError, match='API_DB_POOL_MAX doit'):
            PostgresConfig.from_env(env)


class TestGrafanaConfig:
    def test_valid(self, env):
        config = GrafanaConfig.from_env(env)
        assert config.url == 'http://grafana.test:3000'
        assert config.datasources == ('projet-traces', 'projet-prefect', 'projet-mlflow')
        assert 'secret-admin' not in repr(config)

    def test_password_required_without_token(self, env):
        del env['GRAFANA_ADMIN_PASSWORD']
        with pytest.raises(ConfigError, match='GRAFANA_ADMIN_PASSWORD'):
            GrafanaConfig.from_env(env)

    def test_token_replaces_password(self, env):
        del env['GRAFANA_ADMIN_PASSWORD']
        env['GRAFANA_TOKEN'] = 'glsa_x'
        assert GrafanaConfig.from_env(env).token == 'glsa_x'

    def test_custom_datasources(self, env):
        env['GRAFANA_DATASOURCE_UIDS'] = ' a , b ,'
        assert GrafanaConfig.from_env(env).datasources == ('a', 'b')


def test_process_env_overrides_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / '.env'
    env_file.write_text('MLFLOW_TRACKING_URI=http://from-file:5000\nAUTRE=fichier\n', encoding='utf-8')
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'http://from-process:5000')
    values = load_env(env_file)
    assert values['MLFLOW_TRACKING_URI'] == 'http://from-process:5000'
    assert values['AUTRE'] == 'fichier'


def test_missing_dotenv_is_fine(tmp_path):
    assert isinstance(load_env(tmp_path / 'absent.env'), dict)


def test_env_example_is_complete():
    """.env.example fournit tout ce que chaque connecteur exige."""
    from dotenv import dotenv_values

    from connectors.config import ROOT

    example = {k: v for k, v in dotenv_values(ROOT / '.env.example').items() if v is not None}
    for config_cls in (MlflowConfig, PrefectConfig, PostgresConfig, GrafanaConfig):
        config_cls.from_env(example)
