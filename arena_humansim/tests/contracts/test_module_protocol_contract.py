from __future__ import annotations

import inspect
from collections.abc import Iterable

from arena_humansim.agents.base import BaseAgent, Module, TickPhase, VectorizedModule
from arena_humansim.pool import AgentPool


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


def _declared_methods(proto: type) -> set[str]:
    return {k for k, v in proto.__dict__.items() if not k.startswith("_") and callable(v)}


def test_protocol_accepts_compliant_impl() -> None:
    assert isinstance(_DummyModule(), Module)
    assert isinstance(_DummyVectorized(), VectorizedModule)


def test_module_protocol_attrs() -> None:
    assert _declared_methods(Module) == {"phase", "step_batch"}


def test_vectorized_module_protocol_attrs() -> None:
    assert _declared_methods(VectorizedModule) == {"phase", "step_pool"}


def test_module_rejects_empty() -> None:
    assert not isinstance(_Empty(), Module)


def test_module_rejects_only_phase() -> None:
    assert not isinstance(_OnlyPhase(), Module)


def test_module_rejects_only_step_batch() -> None:
    assert not isinstance(_OnlyStepBatch(), Module)


def test_vectorized_module_rejects_empty() -> None:
    assert not isinstance(_Empty(), VectorizedModule)


def test_vectorized_module_rejects_only_phase() -> None:
    assert not isinstance(_OnlyPhase(), VectorizedModule)


def test_vectorized_module_rejects_missing_step_pool() -> None:
    assert not isinstance(_DummyModule(), VectorizedModule)


def test_step_batch_signature_arity() -> None:
    params = list(inspect.signature(_DummyModule.step_batch).parameters.values())
    assert [p.name for p in params] == ["self", "agents", "dt"]


def test_step_pool_signature_arity() -> None:
    params = list(inspect.signature(_DummyVectorized.step_pool).parameters.values())
    assert [p.name for p in params] == ["self", "pool", "n", "dt"]


def test_tick_phase_values_distinct() -> None:
    values = {TickPhase.SENSE, TickPhase.PLAN, TickPhase.ACT}
    assert len(values) == 3
