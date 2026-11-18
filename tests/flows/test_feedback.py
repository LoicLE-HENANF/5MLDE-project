"""Règle de retour arrière automatique du flow feedback."""

from datetime import date, timedelta

from flows.feedback import ROLLBACK_AFTER_DAYS, should_roll_back

DAY = date(2026, 11, 15)


def _history(*degraded: bool, gap: int | None = None) -> list[dict]:
    rows = []
    day = DAY
    for i, flag in enumerate(degraded):
        rows.append({'day': day, 'degraded': flag})
        day -= timedelta(days=2 if gap == i else 1)
    return rows


def test_rolls_back_after_consecutive_degraded_days():
    assert should_roll_back(_history(*[True] * ROLLBACK_AFTER_DAYS))


def test_keeps_champion_when_one_recent_day_is_healthy():
    assert not should_roll_back(_history(True, False, True))


def test_keeps_champion_with_too_short_history():
    assert not should_roll_back(_history(*[True] * (ROLLBACK_AFTER_DAYS - 1)))


def test_missing_day_breaks_the_streak():
    assert not should_roll_back(_history(True, True, True, gap=0))
