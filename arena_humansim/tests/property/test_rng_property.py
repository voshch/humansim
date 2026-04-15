from __future__ import annotations

"""RNG substream invariants."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from arena_humansim.utils.rng import RNG


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=100)
def test_same_seed_deterministic(seed: int) -> None:
    """Same seed yields identical substream draws."""
    a = RNG(seed)
    b = RNG(seed)
    va = a.get_substream("module").random(10)
    vb = b.get_substream("module").random(10)
    assert np.array_equal(va, vb)


@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    name_a=st.text(min_size=1, max_size=20),
    name_b=st.text(min_size=1, max_size=20),
)
@settings(max_examples=100)
def test_distinct_names_distinct_streams(seed: int, name_a: str, name_b: str) -> None:
    """Distinct substream names yield distinct draws under the same seed."""
    if name_a == name_b:
        return
    r = RNG(seed)
    va = r.get_substream(name_a).random(10)
    vb = r.get_substream(name_b).random(10)
    assert not np.array_equal(va, vb)
