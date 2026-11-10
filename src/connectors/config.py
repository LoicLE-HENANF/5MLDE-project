"""Configuration des connecteurs, lue depuis l'environnement (et le fichier .env à la racine).

Les adresses des serveurs n'ont PAS de valeur par défaut : une variable absente lève ConfigError
au lieu de basculer silencieusement sur un ./mlruns local (MLflow) ou un serveur éphémère (Prefect).
Toutes les erreurs d'une section sont signalées d'un coup.

Chaque service se configure séparément (MlflowConfig.from_env(env)...) : l'API n'a pas besoin
des identifiants Grafana pour démarrer.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

from connectors.errors import ConfigError

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / '.env'


def load_env(env_file: Path | None = ENV_FILE) -> dict[str, str]:
    """Variables du fichier .env (si présent), écrasées par celles du processus."""
    values: dict[str, str] = {}
    if env_file is not None and env_file.is_file():
        values.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
    values.update(os.environ)
    return values


class _Reader:
    """Lit des variables en accumulant les erreurs."""

    def __init__(self, env: Mapping[str, str] | None) -> None:
        self.env = load_env() if env is None else env
        self.errors: list[str] = []

    def present(self, name: str) -> bool:
        return bool(str(self.env.get(name, '')).strip())

    def text(self, name: str, default: str | None = None) -> str:
        value = str(self.env.get(name, '')).strip()
        if value:
            return value
        if default is None:
            self.errors.append(f'{name} : variable absente (voir .env.example)')
            return ''
        return default

    def url(self, name: str, example: str) -> str:
        value = self.text(name)
        if value:
            parts = urlsplit(value)
            if parts.scheme not in ('http', 'https') or not parts.hostname:
                self.errors.append(f'{name}={value!r} : URL http(s) attendue (ex. {example})')
        return value.rstrip('/')

    def integer(self, name: str, default: int, minimum: int = 1, maximum: int | None = None) -> int:
        raw = self.text(name, str(default))
        try:
            value = int(raw)
        except ValueError:
            self.errors.append(f'{name}={raw!r} : entier attendu')
            return default
        if value < minimum or (maximum is not None and value > maximum):
            bound = f'entre {minimum} et {maximum}' if maximum is not None else f'>= {minimum}'
            self.errors.append(f'{name}={value} : valeur {bound} attendue')
        return value

    def finish(self, section: str) -> None:
        if self.errors:
            details = '\n  - '.join(self.errors)
            raise ConfigError(f'Configuration {section} invalide :\n  - {details}')


@dataclass(frozen=True)
class MlflowConfig:
    tracking_uri: str
    registry_uri: str
    experiment: str

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> MlflowConfig:
        reader = _Reader(env)
        tracking = reader.url('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        registry = (
            reader.url('MLFLOW_REGISTRY_URI', 'http://localhost:5000')
            if reader.present('MLFLOW_REGISTRY_URI')
            else tracking
        )
        experiment = reader.text('MLFLOW_EXPERIMENT_NAME', '5mlde-project')
        reader.finish('MLflow')
        return cls(tracking_uri=tracking, registry_uri=registry, experiment=experiment)


@dataclass(frozen=True)
class PrefectConfig:
    api_url: str
    ui_url: str

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> PrefectConfig:
        reader = _Reader(env)
        api_url = reader.url('PREFECT_API_URL', 'http://localhost:4200/api')
        if api_url and not api_url.endswith('/api'):
            reader.errors.append(f'PREFECT_API_URL={api_url!r} : doit se terminer par /api')
        ui_url = (
            reader.url('PREFECT_UI_URL', 'http://localhost:4200')
            if reader.present('PREFECT_UI_URL')
            else api_url.removesuffix('/api')
        )
        reader.finish('Prefect')
        return cls(api_url=api_url, ui_url=ui_url)


@dataclass(frozen=True)
class PostgresConfig:
    """Connexion applicative à la base mlops (rôle api_writer, schéma tracing)."""

    host: str
    port: int
    dbname: str
    user: str
    password: str = field(repr=False)
    pool_min: int = 1
    pool_max: int = 5
    timeout_s: int = 5

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> PostgresConfig:
        reader = _Reader(env)
        values = dict(
            host=reader.text('POSTGRES_HOST', 'localhost'),
            port=reader.integer('POSTGRES_PORT', 5432, maximum=65535),
            dbname=reader.text('API_DB_NAME', 'mlops'),
            user=reader.text('API_DB_USER', 'api_writer'),
            password=reader.text('API_DB_PASSWORD'),
            pool_min=reader.integer('API_DB_POOL_MIN', 1, minimum=0),
            pool_max=reader.integer('API_DB_POOL_MAX', 5),
            timeout_s=reader.integer('API_DB_TIMEOUT_S', 5),
        )
        if values['pool_max'] < values['pool_min']:
            reader.errors.append('API_DB_POOL_MAX doit être >= API_DB_POOL_MIN')
        reader.finish('PostgreSQL')
        return cls(**values)

    @property
    def target(self) -> str:
        return f'{self.user}@{self.host}:{self.port}/{self.dbname}'


@dataclass(frozen=True)
class GrafanaConfig:
    """API HTTP de Grafana : jeton de compte de service (GRAFANA_TOKEN) sinon compte admin."""

    url: str
    user: str
    password: str = field(repr=False)
    token: str = field(default='', repr=False)
    datasources: tuple[str, ...] = ('projet-traces', 'projet-prefect', 'projet-mlflow')

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> GrafanaConfig:
        reader = _Reader(env)
        url = reader.url('GRAFANA_URL', 'http://localhost:3000')
        token = reader.text('GRAFANA_TOKEN', '')
        user = reader.text('GRAFANA_ADMIN_USER', 'admin')
        password = reader.text('GRAFANA_ADMIN_PASSWORD', '' if token else None)
        raw = reader.text('GRAFANA_DATASOURCE_UIDS', ','.join(cls.datasources))
        datasources = tuple(uid.strip() for uid in raw.split(',') if uid.strip())
        reader.finish('Grafana')
        return cls(url=url, user=user, password=password, token=token, datasources=datasources)
