from __future__ import annotations

import pytest

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
