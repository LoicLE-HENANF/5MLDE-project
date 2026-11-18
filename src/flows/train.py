"""Flow `train-volatility` : réentraînement planifié du modèle `volatilite` puis règle de promotion.

Même chaîne que `python -m training.volatility --registre` : jeu de données (Parquet + bougies
récentes), walk-forward, run MLflow (métriques, graphiques, provenance), puis décision du registre
(niveaux 1 et 2, comparaison au champion). Si le code exécuté n'est pas publié (tag git exact,
installation non éditable), le run MLflow est conservé mais rien n'est enregistré au registre.

    uv run --no-editable python -m flows.train
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from math import isfinite
from typing import Any

from prefect import flow, task

from connectors.config import MlflowConfig, PostgresConfig, load_env
from connectors.mlflow import MlflowConnector
from connectors.postgres import PostgresConnector
from connectors.prefect import get_logger, publish_markdown, run_context, task_options
from flows.system import stopped
from lineage import provenance
from training.dataset import prepare_return_dataset, prepare_volatility_dataset
from training.returns import (
    MODEL_NAME as RETURN_MODEL_NAME,
)
from training.returns import (
    TrainingResult as ReturnTrainingResult,
)
from training.returns import (
    fit_return_model,
)
from training.returns import (
    log_training as log_return_training,
)
from training.returns import (
    promote as promote_return,
)
from training.volatility import (
    HORIZON_HOURS,
    MODEL_NAME,
    TrainingResult,
    fit_volatility_model,
    load_candles,
    log_training,
    promote,
)

FLOW_NAME = 'train-volatility'
RETURN_FLOW_NAME = 'train-returns'


@task(name='dataset', **task_options())
def build_dataset(env: dict[str, str], as_of: datetime | None) -> Any:
    log = get_logger()
    candles = load_candles(env)
    dataset = prepare_volatility_dataset(candles, hours=HORIZON_HOURS, as_of=as_of)
    log.info(
        'jeu de données : %d lignes, %d symboles, du %s au %s (as-of %s)',
        len(dataset.frame),
        dataset.frame['symbol'].nunique(),
        f'{dataset.frame["open_time"].min():%Y-%m-%d}',
        f'{dataset.frame["open_time"].max():%Y-%m-%d}',
        f'{dataset.as_of:%Y-%m-%d %H:%M}',
    )
    return dataset


@task(name='fit', **task_options())
def fit(dataset: Any) -> TrainingResult:
    log = get_logger()
    result = fit_volatility_model(dataset)
    log.info('%d folds walk-forward ; candidat retenu : %s', len(result.evaluation.folds), result.candidate)
    for predictor, metrics in result.holdout.items():
        log.info(
            '  test final %-12s QLIKE %.4f  MAE log-vol %.4f',
            predictor,
            metrics['qlike'],
            metrics['mae_log_vol'],
        )
    for strategy, metrics in result.strategy.items():
        log.info(
            '  stratégie %-12s Sharpe %.2f  perte max %.2f%%',
            strategy,
            metrics['sharpe'],
            100 * metrics['max_drawdown'],
        )
    return result


@task(name='register', **task_options())
def register(result: TrainingResult, connector: MlflowConnector, info: Any, db: PostgresConnector) -> dict:
    log = get_logger()
    problems = provenance.release_problems(provenance.current().tags())
    if problems:
        log.warning('code non publié, registre ignoré : %s', ' ; '.join(problems))
        return {'decision': 'ignorée', 'raisons': problems, 'version': None}
    decision, version = promote(result, connector, info, record=db.record_model_event)
    log.info('registre : %s', decision.summary())
    if version is not None:
        log.info('%s version %s -> %s', MODEL_NAME, version.version, decision.alias)
    return {
        'decision': decision.alias or 'rejetée',
        'raisons': list(decision.reasons),
        'version': None if version is None else int(version.version),
    }


def _report(result: TrainingResult, run_url: str, registry: dict) -> str:
    lines = [f'Run MLflow : {run_url}', '', '| Prédicteur | QLIKE | MAE log-vol |', '|---|---|---|']
    for predictor, metrics in result.holdout.items():
        lines.append(f'| {predictor} | {metrics["qlike"]:.4f} | {metrics["mae_log_vol"]:.4f} |')
    lines += ['', f'Registre : **{registry["decision"]}** ({" ; ".join(registry["raisons"])})']
    return '\n'.join(lines)


@flow(name=FLOW_NAME, log_prints=True)
def train_volatility(as_of: datetime | None = None) -> dict[str, Any]:
    env = load_env()
    started = datetime.now(UTC)
    flow_run_id = run_context().get('flow_run_id')
    connector = MlflowConnector(MlflowConfig.from_env(env))
    with PostgresConnector(PostgresConfig.from_env(env)) as db:
        if (halted := stopped(db, FLOW_NAME, started, flow_run_id)) is not None:
            return halted
        code = provenance.current()
        get_logger().info(
            'code %s (git %s, modifié : %s, éditable : %s)',
            code.version,
            code.git_ref,
            code.dirty,
            code.editable,
        )
        dataset = build_dataset(env, as_of)
        result = fit(dataset)
        run_id, info = log_training(result, connector, run_name='train-volatilite-planifie')
        get_logger().info('run MLflow : %s', connector.run_url(run_id))
        registry = register(result, connector, info, db)
        summary = {
            'candidat': result.candidate,
            'qlike': result.holdout[result.candidate]['qlike'],
            'bat_references': result.beats_baselines(),
            'bat_persistance': result.beats_persistence_strategy(),
            'registre': registry,
        }
        db.record_pipeline_run(
            pipeline=FLOW_NAME,
            status='success',
            started_at=started,
            ended_at=datetime.now(UTC),
            prefect_flow_run_id=flow_run_id,
            mlflow_run_id=run_id,
            details=summary,
        )
    get_logger().info('bilan : %s', summary)
    publish_markdown(
        _report(result, connector.run_url(run_id), registry),
        key='entrainement',
        description='Entraînement planifié',
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Entraîne un modèle puis applique sa règle de promotion.')
    parser.add_argument(
        '--model',
        choices=(MODEL_NAME, RETURN_MODEL_NAME),
        default=MODEL_NAME,
        help='modèle à entraîner',
    )
    parser.add_argument(
        '--as-of', type=datetime.fromisoformat, default=None, help='instant de référence (UTC)'
    )
    args = parser.parse_args(argv)
    train = train_returns if args.model == RETURN_MODEL_NAME else train_volatility
    print(train(as_of=args.as_of))
    return 0


@task(name='dataset-rendement', **task_options())
def build_returns_dataset(env: dict[str, str], as_of: datetime | None) -> Any:
    log = get_logger()
    dataset = prepare_return_dataset(load_candles(env), as_of=as_of)
    log.info(
        'jeu de données rendement : %d lignes, %d symboles, du %s au %s (as-of %s)',
        len(dataset.frame),
        dataset.frame['symbol'].nunique(),
        f'{dataset.frame["open_time"].min():%Y-%m-%d}',
        f'{dataset.frame["open_time"].max():%Y-%m-%d}',
        f'{dataset.as_of:%Y-%m-%d %H:%M}',
    )
    return dataset


@task(name='fit-rendement', **task_options())
def fit_returns(dataset: Any) -> ReturnTrainingResult:
    log = get_logger()
    result = fit_return_model(dataset)
    log.info('%d folds walk-forward ; candidat retenu : %s', len(result.evaluation.folds), result.candidate)
    for predictor, metrics in result.holdout.items():
        log.info(
            '  test final %-12s IC %.4f  ICIR %.4f',
            predictor,
            metrics['ic_mean'],
            metrics['icir'],
        )
    for strategy, metrics in result.strategy.items():
        log.info(
            '  stratégie %-28s Sharpe %.2f  rendement %.2f%%',
            strategy,
            metrics['sharpe'],
            100 * metrics['total_return'],
        )
    return result


@task(name='register-rendement', **task_options())
def register_returns(
    result: ReturnTrainingResult, connector: MlflowConnector, info: Any, db: PostgresConnector
) -> dict:
    log = get_logger()
    problems = provenance.release_problems(provenance.current().tags())
    if problems:
        log.warning('code non publié, registre ignoré : %s', ' ; '.join(problems))
        return {'decision': 'ignorée', 'raisons': problems, 'version': None}
    decision, version = promote_return(result, connector, info, record=db.record_model_event)
    log.info('registre : %s', decision.summary())
    if version is not None:
        log.info('%s version %s -> %s', RETURN_MODEL_NAME, version.version, decision.alias)
    return {
        'decision': decision.alias or 'rejetée',
        'raisons': list(decision.reasons),
        'version': None if version is None else int(version.version),
    }


def _returns_report(result: ReturnTrainingResult, run_url: str, registry: dict) -> str:
    lines = [f'Run MLflow : {run_url}', '', '| Prédicteur | IC | ICIR |', '|---|---:|---:|']
    for predictor, metrics in result.holdout.items():
        lines.append(f'| {predictor} | {metrics["ic_mean"]:.4f} | {metrics["icir"]:.4f} |')
    lines += ['', f'Registre : **{registry["decision"]}** ({" ; ".join(registry["raisons"])})']
    return '\n'.join(lines)


@flow(name=RETURN_FLOW_NAME, log_prints=True)
def train_returns(as_of: datetime | None = None) -> dict[str, Any]:
    env = load_env()
    started = datetime.now(UTC)
    flow_run_id = run_context().get('flow_run_id')
    connector = MlflowConnector(MlflowConfig.from_env(env))
    with PostgresConnector(PostgresConfig.from_env(env)) as db:
        if (halted := stopped(db, RETURN_FLOW_NAME, started, flow_run_id)) is not None:
            return halted
        code = provenance.current()
        get_logger().info(
            'code %s (git %s, modifié : %s, éditable : %s)',
            code.version,
            code.git_ref,
            code.dirty,
            code.editable,
        )
        dataset = build_returns_dataset(env, as_of)
        result = fit_returns(dataset)
        run_id, info = log_return_training(result, connector, run_name='train-rendement-planifie')
        get_logger().info('run MLflow : %s', connector.run_url(run_id))
        registry = register_returns(result, connector, info, db)
        summary = {
            'candidat': result.candidate,
            'ic': _finite_or_none(result.holdout[result.candidate]['ic_mean']),
            'icir': _finite_or_none(result.holdout[result.candidate]['icir']),
            'bat_references': result.beats_baselines(),
            'bat_long_short': result.beats_long_short_strategy(),
            'registre': registry,
        }
        db.record_pipeline_run(
            pipeline=RETURN_FLOW_NAME,
            status='success',
            started_at=started,
            ended_at=datetime.now(UTC),
            prefect_flow_run_id=flow_run_id,
            mlflow_run_id=run_id,
            details=summary,
        )
    get_logger().info('bilan : %s', summary)
    publish_markdown(
        _returns_report(result, connector.run_url(run_id), registry),
        key='entrainement-rendement',
        description='Entraînement planifié du modèle rendement',
    )
    return summary


def _finite_or_none(value: float) -> float | None:
    return value if isfinite(value) else None


if __name__ == '__main__':
    raise SystemExit(main())
