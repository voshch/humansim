from __future__ import annotations

import pytest

from arena_humansim.core.agents.types import AttentionDef, AttentionStepDef, GoToStepDef, Pose3, RelativeRef, RobotRef, StepDef
from arena_humansim.utils.scenario import _structure_manual


def _minimal(extra: dict | None = None) -> dict:
    data: dict = {"name": "t", "simulation": {"seed": 1, "dt": 0.05, "max_ticks": 1}, "modules": {}}
    if extra:
        data.update(extra)
    return data


def test_valid_object_bound_step_with_target_type_loads() -> None:
    data = _minimal(
        {
            "world_objects": [{"object_id": "b1", "type": "bench", "pose": {"x": 0.0, "y": 0.0, "theta": 0.0}, "capacity": 1}],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {
                        "rest": {
                            "steps": {
                                "sit_step": {
                                    "interaction": "SIT_ON",
                                    "target": "bench",
                                }
                            }
                        }
                    },
                }
            },
        }
    )
    scn = _structure_manual(data)
    assert scn.agent_types["walker"].sequences["rest"].steps["sit_step"].target == "bench"


def test_unresolvable_step_target_raises() -> None:
    data = _minimal(
        {
            "world_objects": [{"object_id": "b1", "type": "bench", "pose": {"x": 0.0, "y": 0.0, "theta": 0.0}, "capacity": 1}],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {
                        "rest": {
                            "steps": {
                                "go_atm": {
                                    "interaction": "USE",
                                    "target": "atm",
                                }
                            }
                        }
                    },
                }
            },
        }
    )
    with pytest.raises(ValueError, match="target="):
        _structure_manual(data)


def test_unresolvable_action_target_raises() -> None:
    data = _minimal(
        {
            "world_objects": [],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "actions": {
                        "sit": {
                            "interaction": "SIT_ON",
                            "target": "chair",
                            "satisfies": {"rest": 50.0},
                        }
                    },
                }
            },
        }
    )
    with pytest.raises(ValueError, match="target="):
        _structure_manual(data)


def test_valid_target_object_id_loads() -> None:
    data = _minimal(
        {
            "world_objects": [
                {"object_id": "atm_main", "type": "atm", "pose": {"x": 0.0, "y": 0.0, "theta": 0.0}, "capacity": 1},
                {"object_id": "atm_backup", "type": "atm", "pose": {"x": 5.0, "y": 0.0, "theta": 0.0}, "capacity": 1},
            ],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {
                        "queue": {
                            "steps": {
                                "go_specific_atm": {
                                    "interaction": "QUEUE_USE",
                                    "target": "atm_backup",
                                }
                            }
                        }
                    },
                }
            },
        }
    )
    scn = _structure_manual(data)
    assert scn.agent_types["walker"].sequences["queue"].steps["go_specific_atm"].target == "atm_backup"


def test_none_target_skips_validation() -> None:
    data = _minimal(
        {
            "world_objects": [],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {"idle": {"steps": {"pause": {"duration": {"mean": 1.0, "std": 0.0, "clip_low": 0.1, "clip_high": 5.0}}}}},
                }
            },
        }
    )
    scn = _structure_manual(data)
    assert scn.agent_types["walker"].sequences["idle"].steps["pause"].target is None


def _with_step(fields: dict) -> dict:
    return _minimal(
        {
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {"default": {"steps": {"step": fields}}},
                }
            },
        }
    )


def test_legacy_accept_field_raises() -> None:
    with pytest.raises(ValueError, match="'accept' removed"):
        _structure_manual(_with_step({"accept": True, "interaction": "SIT_ON"}))


def test_legacy_target_object_type_raises() -> None:
    with pytest.raises(ValueError, match="'target_object_type' removed"):
        _structure_manual(_with_step({"target_object_type": "bench"}))


def test_legacy_target_object_id_raises() -> None:
    with pytest.raises(ValueError, match="'target_object_id' removed"):
        _structure_manual(_with_step({"target_object_id": "b1"}))


def test_legacy_target_agent_raises() -> None:
    with pytest.raises(ValueError, match="'target_agent' removed"):
        _structure_manual(_with_step({"target_agent": 2}))


def test_legacy_service_tag_raises() -> None:
    with pytest.raises(ValueError, match="'service_tag' removed"):
        _structure_manual(_with_step({"interaction": "SERVICE", "service_tag": "water"}))


def test_follow_interaction_raises() -> None:
    with pytest.raises(ValueError, match="FOLLOW removed"):
        _structure_manual(_with_step({"interaction": "FOLLOW"}))


def test_offer_true_on_non_service_raises() -> None:
    with pytest.raises(ValueError, match="'offer: true' is not valid"):
        _structure_manual(_with_step({"interaction": "TALK_TO", "offer": True}))


def test_offer_true_on_service_without_target_raises() -> None:
    with pytest.raises(ValueError, match="'target: <tag:str>'"):
        _structure_manual(_with_step({"interaction": "SERVICE", "offer": True}))


def test_object_bound_without_target_raises() -> None:
    with pytest.raises(ValueError, match="requires 'target:"):
        _structure_manual(_with_step({"interaction": "SIT_ON"}))


def test_symmetric_with_target_raises() -> None:
    with pytest.raises(ValueError, match="takes no target"):
        _structure_manual(_with_step({"interaction": "TALK_TO", "target": "someone"}))


def test_block_with_valid_target_agent_loads() -> None:
    data = _minimal(
        {
            "agents": [{"agent_id": 1, "agent_type": "walker"}, {"agent_id": 99, "agent_type": "walker"}],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {"default": {"steps": {"pursue": {"interaction": "BLOCK", "target": 99, "duration": {"mean": 5.0, "std": 0.0, "clip_low": 0.1, "clip_high": 30.0}}}}},
                }
            },
        }
    )
    scn = _structure_manual(data)
    assert scn.agent_types["walker"].sequences["default"].steps["pursue"].target == 99


def test_block_with_unknown_target_agent_raises() -> None:
    data = _minimal(
        {
            "agents": [{"agent_id": 1, "agent_type": "walker"}],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {"default": {"steps": {"pursue": {"interaction": "BLOCK", "target": 99}}}},
                }
            },
        }
    )
    with pytest.raises(ValueError, match="BLOCK requires"):
        _structure_manual(data)


def test_block_with_autonomous_raises() -> None:
    data = _minimal(
        {
            "agents": [{"agent_id": 1, "agent_type": "walker"}, {"agent_id": 99, "agent_type": "walker"}],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {"default": {"steps": {"pursue": {"interaction": "BLOCK", "target": 99, "autonomous": True}}}},
                }
            },
        }
    )
    with pytest.raises(ValueError, match="BLOCK and autonomous"):
        _structure_manual(data)


def test_provider_field_without_offer_raises() -> None:
    with pytest.raises(ValueError, match="provider-side"):
        _structure_manual(_with_step({"interaction": "SERVICE", "target": "water", "max_participants": 3}))


def test_go_to_step_with_both_target_pose_and_target_raises() -> None:
    data = _minimal(
        {
            "world_objects": [{"object_id": "b1", "type": "bench", "pose": {"x": 0.0, "y": 0.0, "theta": 0.0}, "capacity": 1}],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {"default": {"steps": {"walk": {"kind": "go_to", "target_pose": {"x": 1.0, "y": 2.0}, "target": "bench"}}}},
                }
            },
        }
    )
    with pytest.raises(ValueError, match="not both"):
        _structure_manual(data)


def _step(fields: dict) -> StepDef | GoToStepDef | AttentionStepDef:
    return _structure_manual(_with_step(fields)).agent_types["walker"].sequences["default"].steps["step"]


def test_attention_step_structures() -> None:
    step = _step({"kind": "attention", "attention": {"gesture": "point", "at": "bench", "hand": "left", "face": "auto", "hold": "keep", "dwell": 0.5}, "duration": {"mean": 1.5}})
    assert isinstance(step, AttentionStepDef)
    assert step.attention == AttentionDef(gesture="point", at="bench", hand="left", face=None, hold="keep", dwell=0.5)
    assert step.duration is not None
    assert step.duration.mean == 1.5


def test_attention_step_dispatch_without_kind() -> None:
    step = _step({"attention": {"gesture": "point", "at": "bench"}, "duration": 2.0, "patience": 5.0, "on_failure": "skip"})
    assert isinstance(step, AttentionStepDef)
    assert step.on_failure == "skip"


def test_attention_step_defaults() -> None:
    step = _step({"attention": {"gesture": "point", "at": "bench"}})
    assert isinstance(step, AttentionStepDef)
    assert step.attention == AttentionDef(gesture="point", at="bench")
    assert step.attention.hand == "auto"
    assert step.attention.face is None
    assert step.attention.hold == "release"
    assert step.attention.dwell == 1.0


def test_attention_on_wait_step_stays_step_def() -> None:
    step = _step({"duration": 2.0, "interaction": "TALK_TO", "attention": {"gesture": "point", "at": "partner"}})
    assert isinstance(step, StepDef)
    assert step.attention == AttentionDef(gesture="point", at="partner")


def test_attention_on_cancel_step() -> None:
    step = _step({"cancel": True, "attention": {"gesture": "wave", "at": "partner"}})
    assert isinstance(step, StepDef)
    assert step.cancel is True
    assert step.attention is not None


def test_attention_on_go_to_step() -> None:
    step = _step({"kind": "go_to", "target_pose": {"x": 1.0, "y": 2.0}, "attention": {"gesture": "point", "at": ["partner", "ped_1"], "at_z": 1.4, "face": True}})
    assert isinstance(step, GoToStepDef)
    assert step.attention == AttentionDef(gesture="point", at=("partner", "ped_1"), at_z=1.4, face=True)


def test_go_to_step_unknown_field_raises() -> None:
    with pytest.raises(ValueError, match="unknown go_to step fields"):
        _step({"kind": "go_to", "target_pose": {"x": 1.0, "y": 2.0}, "gesture": "point"})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("partner", "partner"),
        ("partners", "partners"),
        ("target", "target"),
        ("goal", "goal"),
        ("robot:bot", RobotRef("bot")),
        ("bench_1", "bench_1"),
        (7, 7),
        ({"x": 1.0, "y": 2.0, "z": 3.0}, Pose3(1.0, 2.0, 3.0)),
        ({"azimuth": 90, "elevation": 10}, RelativeRef(90.0, 10.0, 3.0)),
        ({"azimuth": -45, "elevation": 0, "distance": 1.5}, RelativeRef(-45.0, 0.0, 1.5)),
    ],
)
def test_attention_ref_kinds(raw: object, expected: object) -> None:
    step = _step({"attention": {"gesture": "point", "at": raw}})
    assert isinstance(step, AttentionStepDef)
    assert step.attention.at == expected


def test_attention_at_list_structures_each_ref() -> None:
    step = _step({"attention": {"gesture": "point", "at": ["partner", 3, {"x": 0, "y": 0, "z": 0}]}})
    assert isinstance(step, AttentionStepDef)
    assert step.attention.at == ("partner", 3, Pose3(0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    ("att", "match"),
    [
        ({"gesture": "", "at": "bench"}, "gesture"),
        ({"gesture": 3, "at": "bench"}, "gesture"),
        ({"gesture": "point"}, "at"),
        ({"gesture": "point", "at": []}, "must not be empty"),
        ({"gesture": "point", "at": True}, "bool"),
        ({"gesture": "point", "at": ""}, "non-empty"),
        ({"gesture": "point", "at": "robot:"}, "robot:<name>"),
        ({"gesture": "point", "at": {"x": 1.0, "y": 2.0}}, "x, y, z"),
        ({"gesture": "point", "at": {"azimuth": 1.0}}, "azimuth"),
        ({"gesture": "point", "at": {"azimuth": "a", "elevation": 0}}, "number"),
        ({"gesture": "point", "at": {"azimuth": 0, "elevation": 0, "distance": 0}}, "distance"),
        ({"gesture": "point", "at": 1.5}, "ref must be"),
        ({"gesture": "point", "at": "bench", "hand": "both"}, "hand"),
        ({"gesture": "point", "at": "bench", "face": "maybe"}, "face"),
        ({"gesture": "point", "at": "bench", "hold": "forever"}, "hold"),
        ({"gesture": "point", "at": "bench", "dwell": 0}, "dwell"),
        ({"gesture": "point", "at": "bench", "dwell": "x"}, "dwell"),
        ({"gesture": "point", "at": "bench", "at_z": "high"}, "at_z"),
        ({"gesture": "point", "at": {"x": 1.0, "y": 2.0, "z": 3.0}, "at_z": 0.5}, "at_z"),
        ({"gesture": "point", "at": {"azimuth": 0, "elevation": 0}, "at_z": 0.5}, "at_z"),
        ({"gesture": "point", "at": ["bench", {"azimuth": 0, "elevation": 0}], "at_z": 0.5}, "at_z"),
        ({"gesture": "point", "at": {"azimuth": 0, "elevation": 0}, "face": True}, "face: true"),
        ({"gesture": "point", "at": ["bench", {"azimuth": 0, "elevation": 0}], "face": "true"}, "face: true"),
        ({"gesture": "point", "at": "bench", "release": False}, "unknown attention fields"),
    ],
)
def test_attention_validation(att: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _step({"attention": att})


def test_attention_not_a_mapping_raises() -> None:
    with pytest.raises(ValueError, match="mapping"):
        _step({"attention": "point"})


def test_attention_none_gesture_without_at() -> None:
    step = _step({"attention": {"gesture": "none"}})
    assert isinstance(step, AttentionStepDef)
    assert step.attention.at is None


def test_attention_face_forms() -> None:
    assert _step({"attention": {"gesture": "point", "at": "bench", "face": True}}).attention.face is True
    assert _step({"attention": {"gesture": "point", "at": "bench", "face": "false"}}).attention.face is False
    assert _step({"attention": {"gesture": "point", "at": "bench", "face": "auto"}}).attention.face is None


def test_attention_at_z_numeric_with_entity_ref() -> None:
    assert _step({"attention": {"gesture": "point", "at": "bench", "at_z": 1}}).attention.at_z == 1.0
    assert _step({"attention": {"gesture": "point", "at": ["partner", 3], "at_z": 1.5}}).attention.at_z == 1.5


def test_attention_step_unknown_field_raises() -> None:
    with pytest.raises(ValueError, match="unknown attention step fields"):
        _step({"kind": "attention", "attention": {"gesture": "point", "at": "bench"}, "target": "bench"})


def test_attention_with_step_only_fields_names_them() -> None:
    with pytest.raises(ValueError, match=r"step has attention plus interaction-only fields \['offer', 'until'\], add kind or interaction"):
        _step({"attention": {"gesture": "point", "at": "bench"}, "offer": True, "until": "x"})


def test_attention_on_autonomous_step_raises() -> None:
    with pytest.raises(ValueError, match="not supported on 'autonomous: true' steps"):
        _step({"autonomous": True, "attention": {"gesture": "point", "at": "bench"}})


def test_attention_kind_requires_block() -> None:
    with pytest.raises(ValueError, match="attention step requires"):
        _step({"kind": "attention", "duration": 1.0})


def test_attention_in_actions_raises() -> None:
    data = _minimal(
        {
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "actions": {"show": {"attention": {"gesture": "point", "at": "bench"}}},
                }
            },
        }
    )
    with pytest.raises(ValueError, match="not supported in the autonomous 'actions' library"):
        _structure_manual(data)


def test_unknown_step_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown step kind"):
        _step({"kind": "gesture", "gesture": "point", "at": "bench"})
