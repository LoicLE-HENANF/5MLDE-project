"""Interrupteur marche / arrêt partagé par les flows.

Chaque flow lit l'état du système au démarrage (`tracing.system_changes`, modifié via l'API) ;
s'il est arrêté, il trace un run `cancelled` et s'arrête sans rien faire.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from connectors.postgres import PostgresConnector
from connectors.prefect import get_logger


def stopped(
    db: PostgresConnector, pipeline: str, started: datetime, flow_run_id: str | None
) -> dict[str, Any] | None:
    """Résumé `arrêté` (run tracé `cancelled`) si le système est inactif, None sinon."""
    state = db.system_state()
    if state['active']:
        return None
    details = {'raison': 'système arrêté', 'par': state['changed_by'], 'motif': state['reason']}
    db.record_pipeline_run(
        pipeline=pipeline,
        status='cancelled',
        started_at=started,
        ended_at=datetime.now(UTC),
        prefect_flow_run_id=flow_run_id,
        details=details,
    )
    get_logger().warning(
        '%s : système arrêté par %s (%s), rien à faire', pipeline, state['changed_by'], state['reason']
    )
    return {'statut': 'arrêté', **details}
