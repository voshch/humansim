from __future__ import annotations

"""SimulationParams serialization invariants."""

from hypothesis import given, settings
from hypothesis import strategies as st

from arena_humansim.utils.scenario import SimulationParams
from arena_humansim.utils.scenario_loader import converter


@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    dt=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
    max_ticks=st.integers(min_value=1, max_value=1000),
    bt_tick_interval=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=100)
def test_simulation_params_round_trips(
    seed: int,
    dt: float,
    max_ticks: int,
    bt_tick_interval: int,
) -> None:
    """SimulationParams round-trips through the cattrs converter."""
    params = SimulationParams(
        seed=seed,
        dt=dt,
        max_ticks=max_ticks,
        bt_tick_interval=bt_tick_interval,
    )
    raw = converter.unstructure(params)
    roundtripped = converter.structure(raw, SimulationParams)
    assert roundtripped == params
