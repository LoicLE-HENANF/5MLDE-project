"""Le rejeu n'écrit, ne lit et n'efface que ses propres traces (`mode='replay'`, `*-replay`)."""

from __future__ import annotations

from replay import store


def test_reads_and_writes_are_restricted_to_replay():
    assert "'replay'" in store.PREDICTION_SQL
    for query in (store.PENDING_SQL, store.SCORED_SQL):
        assert "p.mode = 'replay'" in query
    assert "mode = 'replay'" in store.HEALTH_SQL


def test_reset_only_targets_replay_traces():
    for statement in store.RESET_SQL:
        assert "'replay'" in statement or "-replay'" in statement or "'replay-%'" in statement
        assert 'live' not in statement
