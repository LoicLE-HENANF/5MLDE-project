"""Connecteur MLflow : suivi des expériences et registre de modèles (serveur HTTP de la stack).

Usage typique dans une tâche d'entraînement :

    connector = MlflowConnector(MlflowConfig.from_env())
    with connector.start_run('train-rendement', tags=prefect_tags()) as run:
        mlflow.log_params(...)                     # API fluide, déjà pointée vers la stack
        info = connector.log_model(mlflow.sklearn.log_model, sk_model=model)
    connector.register_model(info, 'rendement', alias='challenger')

Chaque run et chaque modèle portent les tags `code.*` (version, commit, empreinte du code et des
dépendances) ; le modèle embarque le paquet `connectors` et ses dépendances épinglées.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import httpx
import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.models.model import ModelInfo

from connectors import provenance
from connectors._http import check_http
from connectors.config import MlflowConfig
from connectors.errors import ConfigError, ProvenanceError
from connectors.provenance import CodeProvenance

PROJECT_TAG = '5mlde-projet'
_NOT_FOUND = {'RESOURCE_DOES_NOT_EXIST', 'INVALID_PARAMETER_VALUE'}


REQUIREMENTS_ARTIFACT = 'provenance/requirements.txt'


class MlflowConnector:
    service = 'MLflow'

    def __init__(
        self, config: MlflowConfig, client: MlflowClient | None = None, http: httpx.Client | None = None
    ) -> None:
        self.config = config
        self.client = client or MlflowClient(
            tracking_uri=config.tracking_uri, registry_uri=config.registry_uri
        )
        self._http = http
        self._experiment_id: str | None = None
        self._run_provenance: dict[str, CodeProvenance] = {}

    # -- santé ---------------------------------------------------------------------------------
    def check(self, timeout: float = 5.0) -> None:
        hint = 'vérifier MLFLOW_TRACKING_URI et que le service mlflow de compose est healthy'
        check_http(
            self.service, f'{self.config.tracking_uri}/health', timeout=timeout, client=self._http, hint=hint
        )
        if self.config.registry_uri != self.config.tracking_uri:
            check_http(
                self.service,
                f'{self.config.registry_uri}/health',
                timeout=timeout,
                client=self._http,
                hint='vérifier MLFLOW_REGISTRY_URI',
            )

    # -- suivi ---------------------------------------------------------------------------------
    def activate(self) -> None:
        """Pointe l'API fluide (mlflow.start_run, log_model, register_model...) vers la stack."""
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_registry_uri(self.config.registry_uri)

    def experiment_id(self) -> str:
        if self._experiment_id is None:
            name = self.config.experiment
            experiment = self.client.get_experiment_by_name(name)
            if experiment is None:
                self._experiment_id = self.client.create_experiment(name, tags={'projet': PROJECT_TAG})
            elif experiment.lifecycle_stage != 'active':
                raise ConfigError(
                    f"L'expérience MLflow {name!r} est supprimée : la restaurer "
                    f'ou changer MLFLOW_EXPERIMENT_NAME'
                )
            else:
                self._experiment_id = experiment.experiment_id
        return self._experiment_id

    @contextmanager
    def start_run(
        self, name: str, *, tags: Mapping[str, str] | None = None, nested: bool = False
    ) -> Iterator[mlflow.ActiveRun]:
        """Run actif (API fluide), tagué avec la provenance du code et ses dépendances épinglées.

        Une exception dans le bloc termine le run en FAILED.
        """
        self.activate()
        code = provenance.current()
        all_tags = {'projet': PROJECT_TAG, **(tags or {}), **code.tags()}
        with mlflow.start_run(
            experiment_id=self.experiment_id(), run_name=name, tags=all_tags, nested=nested
        ) as run:
            mlflow.log_text('\n'.join(code.requirements) + '\n', REQUIREMENTS_ARTIFACT)
            self._run_provenance[run.info.run_id] = code
            try:
                yield run
            finally:
                self._run_provenance.pop(run.info.run_id, None)

    def log_model(self, log_fn: Callable[..., ModelInfo], *, name: str = 'model', **kwargs: Any) -> ModelInfo:
        """Logge un modèle dans le run actif en y embarquant son code et ses dépendances exactes.

        `log_fn` est la fonction de la saveur MLflow, ex. :
            connector.log_model(mlflow.sklearn.log_model, sk_model=model)
            connector.log_model(mlflow.pyfunc.log_model, python_model=Wrapper())
        Le paquet `connectors` est copié dans le modèle (code_paths) et les dépendances sont celles
        installées au moment de l'entraînement (pip_requirements), versions exactes.
        """
        run = mlflow.active_run()
        code = self._run_provenance.get(run.info.run_id) if run else None
        if code is None:
            raise ProvenanceError('log_model doit être appelé dans MlflowConnector.start_run')
        for managed in ('code_paths', 'pip_requirements', 'extra_pip_requirements', 'conda_env'):
            if managed in kwargs:
                raise ValueError(f'{managed} est fixé par le connecteur (provenance du code)')
        return log_fn(
            name=name,
            code_paths=[str(d) for d in code.code_dirs],
            pip_requirements=list(code.requirements),
            tags={**kwargs.pop('tags', {}), **code.tags()},
            **kwargs,
        )

    def run_url(self, run_id: str, experiment_id: str | None = None) -> str:
        experiment = experiment_id or self.experiment_id()
        return f'{self.config.tracking_uri}/#/experiments/{experiment}/runs/{run_id}'

    # -- registre ------------------------------------------------------------------------------
    def register_model(
        self,
        model: ModelInfo,
        name: str,
        *,
        alias: str | None = None,
        tags: Mapping[str, str] | None = None,
        allow_unreleased: bool = False,
    ) -> ModelVersion:
        """Nouvelle version de `name` depuis un modèle loggé par `log_model`.

        La version reçoit les tags de provenance du modèle. Refusé si ce code n'est pas
        identifiable (non commité, installation éditable, provenance absente), sauf
        `allow_unreleased=True` réservé aux essais.
        """
        self.activate()
        code_tags = {
            k: v
            for k, v in self.client.get_logged_model(model.model_id).tags.items()
            if k.startswith(provenance.TAG_PREFIX)
        }
        problems = provenance.release_problems(code_tags)
        if problems and not allow_unreleased:
            raise ProvenanceError(f'Modèle {name!r} non enregistré : ' + ' ; '.join(problems))
        version = mlflow.register_model(model.model_uri, name, tags={**(tags or {}), **code_tags})
        if alias:
            self.set_alias(name, alias, version.version)
        return version

    def set_alias(self, name: str, alias: str, version: str | int) -> None:
        self.client.set_registered_model_alias(name, alias, str(version))

    def get_version(self, name: str, alias: str) -> ModelVersion | None:
        """Version pointée par l'alias, ou None si le modèle ou l'alias n'existe pas."""
        try:
            return self.client.get_model_version_by_alias(name, alias)
        except MlflowException as exc:
            if exc.error_code in _NOT_FOUND:
                return None
            raise

    def load_model(self, name: str, alias: str, loader: Callable[[str], Any] | None = None) -> Any:
        """Charge models:/name@alias (pyfunc par défaut, ex. loader=mlflow.sklearn.load_model)."""
        self.activate()
        return (loader or mlflow.pyfunc.load_model)(f'models:/{name}@{alias}')
