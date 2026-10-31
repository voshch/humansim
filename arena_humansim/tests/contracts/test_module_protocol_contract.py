from __future__ import annotations

from collections.abc import Iterable

import pytest

from arena_humansim.core.agents.base import BaseAgent, Module, TickPhase, VectorizedModule
from arena_humansim.core.pool import AgentPool


class _DummyModule:
    def phase(self) -> TickPhase:
        return TickPhase.PLAN

    def step_batch(self, agents: Iterable[BaseAgent], dt: float) -> None:
        pass


class _DummyVectorized:
    def phase(self) -> TickPhase:
        return TickPhase.ACT

    def step_pool(self, pool: AgentPool, n: int, dt: float) -> None:
        pass


class _Empty:
    pass


class _OnlyPhase:
    def phase(self) -> TickPhase:
        return TickPhase.PLAN


class _OnlyStepBatch:
    def step_batch(self, agents: Iterable[BaseAgent], dt: float) -> None:
        pass


class _OnlyStepPool:
    def step_pool(self, pool: AgentPool, n: int, dt: float) -> None:
        pass


def test_protocol_accepts_compliant_impl() -> None:
    assert isinstance(_DummyModule(), Module)
    assert isinstance(_DummyVectorized(), VectorizedModule)


@pytest.mark.parametrize("impl", [_Empty, _OnlyPhase, _OnlyStepBatch, _DummyVectorized], ids=lambda c: c.__name__)
def test_module_rejects_incomplete_impl(impl: type) -> None:
    assert not isinstance(impl(), Module)


@pytest.mark.parametrize("impl", [_Empty, _OnlyPhase, _OnlyStepPool, _DummyModule], ids=lambda c: c.__name__)
def test_vectorized_module_rejects_incomplete_impl(impl: type) -> None:
    assert not isinstance(impl(), VectorizedModule)
