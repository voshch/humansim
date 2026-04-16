from __future__ import annotations

import ast

import pytest

from arena_humansim.core.agents.types import VarDef
from arena_humansim.utils.scenario import (
    _resolve_var_string,
    _safe_eval,
    _structure_manual,
    _type_check_var,
    _walk_resolve,
    resolve_vars,
)


def _eval(src: str, variables: dict[str, int | float | bool | str] | None = None):
    tree = ast.parse(src, mode="eval")
    return _safe_eval(tree, variables or {})


def test_safe_eval_add() -> None:
    assert _eval("1 + 2") == 3


def test_safe_eval_sub() -> None:
    assert _eval("5 - 3") == 2


def test_safe_eval_mul() -> None:
    assert _eval("4 * 2.5") == 10.0


def test_safe_eval_div() -> None:
    assert _eval("10 / 4") == 2.5


def test_safe_eval_unary_neg() -> None:
    assert _eval("-3") == -3


def test_safe_eval_unary_pos() -> None:
    assert _eval("+4") == 4


def test_safe_eval_name_lookup() -> None:
    assert _eval("x + 1", {"x": 10}) == 11


def test_safe_eval_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown variable"):
        _eval("missing + 1")


def test_safe_eval_unsupported_binop_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported operator"):
        _eval("3 % 2")


def test_safe_eval_unsupported_unary_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported unary operator"):
        _eval("not True")


def test_safe_eval_string_constant_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported constant"):
        _eval("'hello'")


def test_safe_eval_unsupported_node_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported AST node"):
        _eval("[1, 2]")


def test_resolve_var_string_pure_ref_returns_typed_int() -> None:
    result = _resolve_var_string("${x}", {"x": 7})
    assert result == 7
    assert isinstance(result, int)


def test_resolve_var_string_pure_expr_returns_typed_float() -> None:
    result = _resolve_var_string("${x * 0.5}", {"x": 4})
    assert result == 2.0
    assert isinstance(result, float)


def test_resolve_var_string_embedded_returns_str() -> None:
    result = _resolve_var_string("foo_${x}_${y+1}", {"x": 3, "y": 2})
    assert result == "foo_3_3"
    assert isinstance(result, str)


def test_resolve_var_string_no_interpolation_passthrough() -> None:
    assert _resolve_var_string("plain", {}) == "plain"


def test_resolve_var_string_embedded_uses_name_branch() -> None:
    result = _resolve_var_string("a_${x}_b_${y}", {"x": 1, "y": 2})
    assert result == "a_1_b_2"


def test_type_check_var_int_mismatch_raises() -> None:
    vdef = VarDef(type="int", default=0)
    with pytest.raises(TypeError, match="expected type int"):
        _type_check_var("n", 1.5, vdef)


def test_type_check_var_str_mismatch_raises() -> None:
    vdef = VarDef(type="str", default="x")
    with pytest.raises(TypeError, match="expected type str"):
        _type_check_var("s", 5, vdef)


def test_type_check_var_float_accepts_int() -> None:
    vdef = VarDef(type="float", default=0.0)
    _type_check_var("f", 3, vdef)


def test_type_check_var_unknown_type_noop() -> None:
    vdef = VarDef(type="weird", default=0)
    _type_check_var("x", "anything", vdef)


def test_resolve_vars_applies_defaults() -> None:
    var_defs = {"x": VarDef(type="int", default=5)}
    out = resolve_vars({"v": "${x}"}, var_defs)
    assert out == {"v": 5}


def test_resolve_vars_applies_overrides() -> None:
    var_defs = {"x": VarDef(type="int", default=5)}
    out = resolve_vars({"v": "${x}"}, var_defs, overrides={"x": 9})
    assert out == {"v": 9}


def test_resolve_vars_unknown_override_raises() -> None:
    var_defs = {"x": VarDef(type="int", default=1)}
    with pytest.raises(ValueError, match="unknown variable"):
        resolve_vars({}, var_defs, overrides={"ghost": 1})


def test_resolve_vars_type_mismatch_raises() -> None:
    var_defs = {"x": VarDef(type="int", default=1)}
    with pytest.raises(TypeError):
        resolve_vars({}, var_defs, overrides={"x": "nope"})


def test_resolve_vars_below_min_raises() -> None:
    var_defs = {"x": VarDef(type="int", default=5, min=0, max=10)}
    with pytest.raises(ValueError, match="below minimum"):
        resolve_vars({}, var_defs, overrides={"x": -1})


def test_resolve_vars_above_max_raises() -> None:
    var_defs = {"x": VarDef(type="int", default=5, min=0, max=10)}
    with pytest.raises(ValueError, match="above maximum"):
        resolve_vars({}, var_defs, overrides={"x": 99})


def test_resolve_vars_bounds_skipped_for_non_numeric() -> None:
    var_defs = {"s": VarDef(type="str", default="hi")}
    out = resolve_vars({"v": "${s}"}, var_defs)
    assert out == {"v": "hi"}


def test_walk_resolve_descends_list_and_dict() -> None:
    variables: dict[str, int | float | bool | str] = {"x": 2, "y": 3}
    raw = {
        "list": ["${x}", "lit", {"nested": "${y+1}"}],
        "dict": {"a": "${x}", "b": "pre_${y}_post"},
        "scalar": 42,
    }
    out = _walk_resolve(raw, variables)
    assert out == {
        "list": [2, "lit", {"nested": 4}],
        "dict": {"a": 2, "b": "pre_3_post"},
        "scalar": 42,
    }


def test_walk_resolve_passthrough_non_collection() -> None:
    assert _walk_resolve(None, {}) is None
    assert _walk_resolve(7, {}) == 7


def test_structure_manual_expands_vars_into_desired_velocity() -> None:
    data = {
        "name": "s",
        "agent_types": {
            "foo": {
                "vars": {"x": {"type": "int", "default": 2}},
                "desired_velocity": "${x*0.5}",
            },
        },
    }
    scn = _structure_manual(data)
    assert scn.agent_types["foo"].desired_velocity.mean == 1.0


def test_structure_manual_vars_honors_override() -> None:
    data = {
        "name": "s",
        "agent_types": {
            "foo": {
                "vars": {"x": {"type": "int", "default": 2}},
                "desired_velocity": "${x*0.5}",
            },
        },
    }
    scn = _structure_manual(data, var_overrides={"x": 4})
    assert scn.agent_types["foo"].desired_velocity.mean == 2.0


def test_structure_manual_no_vars_leaves_literal() -> None:
    data = {
        "name": "s",
        "agent_types": {
            "foo": {"desired_velocity": 1.7},
        },
    }
    scn = _structure_manual(data)
    assert scn.agent_types["foo"].desired_velocity.mean == 1.7
