"""Plan du test de charge : paliers, SLO, point de rupture et recommandation."""

from __future__ import annotations

import pytest

from loadtest.plan import (
    CONFIGS,
    PEAK,
    TARGET_BOTS,
    Config,
    ConfigResult,
    Measure,
    Slo,
    Step,
    breaking_point,
    ramp,
    recommend,
)

SLO = Slo(p95_ms=200, max_error_rate=0.01)


def _measure(bots=100, p95=50.0, requests=1000, failures=0) -> Measure:
    step = Step(bots=bots, analysts=20, operators=2, spawn_rate=100, duration_s=20)
    return Measure(step, 20.0, requests, failures, 10.0, p95, p95, p95)


def _result(workers, cpus, *, peak_ok=True, held=0) -> ConfigResult:
    """Rampe de 100 en 100 bots tenue jusqu'à `held` bots, puis un palier en échec."""
    steps = [_measure(bots) for bots in range(100, held + 1, 100)]
    steps.append(_measure(held + 100, p95=900.0))
    return ConfigResult(
        Config(workers, cpus), _measure(TARGET_BOTS, 50.0 if peak_ok else 900.0), tuple(steps)
    )


def test_step_counts_users_and_rejects_empty_or_invalid():
    assert PEAK.users == TARGET_BOTS + 22
    for kwargs in ({'bots': 0, 'analysts': 0, 'operators': 0}, {'bots': -1}, {'spawn_rate': 0}):
        with pytest.raises(ValueError):
            Step(**{'bots': 1, 'analysts': 0, 'operators': 0, 'spawn_rate': 1, 'duration_s': 1, **kwargs})


def test_ramp_is_increasing_and_bounded():
    bots = [step.bots for step in ramp(100, 100, 350)]
    assert bots == [100, 200, 300]
    with pytest.raises(ValueError):
        list(ramp(0))


def test_configs_are_distinct_and_valid():
    assert len(set(CONFIGS)) == len(CONFIGS)
    with pytest.raises(ValueError):
        Config(0, 1.0)


@pytest.mark.parametrize(
    ('measure', 'ok'),
    [
        (_measure(p95=199.0), True),
        (_measure(p95=200.0), False),  # seuil strict
        (_measure(failures=10), False),  # 1 % d'erreurs
        (_measure(failures=9), True),
        (_measure(requests=0, p95=None), False),  # rien n'a abouti
    ],
)
def test_slo(measure, ok):
    assert measure.meets(SLO) is ok


def test_measure_rates():
    measure = _measure(requests=400, failures=4)
    assert measure.rps == pytest.approx(20.0)
    assert measure.error_rate == pytest.approx(0.01)


def test_breaking_point_is_last_step_before_first_failure():
    steps = [_measure(100), _measure(200), _measure(300, p95=500.0), _measure(400)]
    assert breaking_point(steps, SLO).step.bots == 200
    assert breaking_point([_measure(100, p95=500.0)], SLO) is None


def test_recommend_picks_the_cheapest_config_with_margin():
    results = [
        _result(4, 1.0, held=1500),
        _result(1, 0.5, held=300),  # marge insuffisante (400 bots demandés)
        _result(2, 0.5, held=500),
        _result(1, 1.0, held=900),
    ]
    choice = recommend(results, SLO)
    assert choice.config == Config(2, 0.5)
    assert '500 bots' in choice.reason


def test_recommend_requires_the_peak():
    choice = recommend([_result(1, 0.5, peak_ok=False, held=800), _result(1, 1.0, held=400)], SLO)
    assert choice.config == Config(1, 1.0)


def test_recommend_without_any_suitable_config_names_the_most_robust():
    choice = recommend([_result(1, 0.5, held=100), _result(2, 1.0, held=300)], SLO)
    assert choice.config is None
    assert '2w / 1 CPU' in choice.reason
    with pytest.raises(ValueError):
        recommend([], SLO)
