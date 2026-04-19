from __future__ import annotations

import pytest

from arena_humansim.utils.scenario import _structure_manual


def _minimal(extra: dict | None = None) -> dict:
    data: dict = {"name": "t", "simulation": {"seed": 1, "dt": 0.05, "max_ticks": 1}, "modules": {}}
    if extra:
        data.update(extra)
    return data


def test_valid_scenario_with_target_object_type_loads() -> None:
    data = _minimal(
        {
            "world_objects": [
                {"object_id": "b1", "type": "bench", "pose": {"x": 0.0, "y": 0.0, "theta": 0.0}, "capacity": 1}
            ],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {
                        "stroll": {
                            "steps": {
                                "go_bench": {
                                    "target_object_type": "bench",
                                }
                            }
                        }
                    },
                }
            },
        }
    )
    scn = _structure_manual(data)
    assert scn.agent_types["walker"].sequences["stroll"].steps["go_bench"].target_object_type == "bench"


def test_unresolvable_step_target_object_type_raises() -> None:
    data = _minimal(
        {
            "world_objects": [
                {"object_id": "b1", "type": "bench", "pose": {"x": 0.0, "y": 0.0, "theta": 0.0}, "capacity": 1}
            ],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {
                        "stroll": {
                            "steps": {
                                "go_atm": {
                                    "target_object_type": "atm",
                                }
                            }
                        }
                    },
                }
            },
        }
    )
    with pytest.raises(ValueError, match="target_object_type="):
        _structure_manual(data)


def test_unresolvable_action_target_object_type_raises() -> None:
    data = _minimal(
        {
            "world_objects": [],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "actions": {
                        "sit": {
                            "target_object_type": "chair",
                            "satisfies": {"rest": 50.0},
                        }
                    },
                }
            },
        }
    )
    with pytest.raises(ValueError, match="target_object_type="):
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
                                    "target_object_id": "atm_backup",
                                }
                            }
                        }
                    },
                }
            },
        }
    )
    scn = _structure_manual(data)
    assert scn.agent_types["walker"].sequences["queue"].steps["go_specific_atm"].target_object_id == "atm_backup"


def test_unresolvable_step_target_object_id_raises() -> None:
    data = _minimal(
        {
            "world_objects": [
                {"object_id": "atm_main", "type": "atm", "pose": {"x": 0.0, "y": 0.0, "theta": 0.0}, "capacity": 1}
            ],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {
                        "queue": {
                            "steps": {
                                "go_missing": {
                                    "target_object_id": "atm_does_not_exist",
                                }
                            }
                        }
                    },
                }
            },
        }
    )
    with pytest.raises(ValueError, match="target_object_id="):
        _structure_manual(data)


def test_both_target_object_id_and_type_raises() -> None:
    data = _minimal(
        {
            "world_objects": [
                {"object_id": "atm_main", "type": "atm", "pose": {"x": 0.0, "y": 0.0, "theta": 0.0}, "capacity": 1}
            ],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {
                        "queue": {
                            "steps": {
                                "go_atm": {
                                    "target_object_id": "atm_main",
                                    "target_object_type": "atm",
                                }
                            }
                        }
                    },
                }
            },
        }
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _structure_manual(data)


def test_none_target_object_type_skips_validation() -> None:
    data = _minimal(
        {
            "world_objects": [],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {
                        "idle": {
                            "steps": {
                                "pause": {"duration": {"mean": 1.0, "std": 0.0, "clip_low": 0.1, "clip_high": 5.0}}
                            }
                        }
                    },
                }
            },
        }
    )
    scn = _structure_manual(data)
    assert scn.agent_types["walker"].sequences["idle"].steps["pause"].target_object_type is None


def _with_accept_step(fields: dict) -> dict:
    return _minimal(
        {
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {
                        "default": {
                            "steps": {"step": fields}
                        }
                    },
                }
            },
        }
    )


def test_accept_without_interaction_raises() -> None:
    with pytest.raises(ValueError, match="accept=true requires interaction"):
        _structure_manual(_with_accept_step({"accept": True}))


def test_accept_with_target_object_raises() -> None:
    data = _minimal(
        {
            "world_objects": [{"object_id": "b1", "type": "bench", "pose": {"x": 0.0, "y": 0.0, "theta": 0.0}, "capacity": 1}],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {"default": {"steps": {"step": {"accept": True, "interaction": "SIT_ON", "target_object_type": "bench"}}}},
                }
            },
        }
    )
    with pytest.raises(ValueError, match="accept steps cannot also target an object"):
        _structure_manual(data)


def test_accept_with_autonomous_raises() -> None:
    with pytest.raises(ValueError, match="accept and autonomous are mutually exclusive"):
        _structure_manual(_with_accept_step({"accept": True, "interaction": "TALK_TO", "autonomous": True}))


def test_accept_with_target_agent_raises() -> None:
    data = _minimal(
        {
            "agents": [{"agent_id": 1, "agent_type": "walker"}, {"agent_id": 2, "agent_type": "walker"}],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {"default": {"steps": {"step": {"accept": True, "interaction": "TALK_TO", "target_agent": 2}}}},
                }
            },
        }
    )
    with pytest.raises(ValueError, match="accept and target_agent are mutually exclusive"):
        _structure_manual(data)


def test_service_tag_without_accept_raises() -> None:
    with pytest.raises(ValueError, match="service_tag requires accept=true"):
        _structure_manual(_with_accept_step({"interaction": "SERVICE", "service_tag": "water"}))


def test_block_with_valid_target_agent_loads() -> None:
    data = _minimal(
        {
            "agents": [{"agent_id": 1, "agent_type": "walker"}, {"agent_id": 99, "agent_type": "walker"}],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {"default": {"steps": {"pursue": {"target_agent": 99, "duration": {"mean": 5.0, "std": 0.0, "clip_low": 0.1, "clip_high": 30.0}}}}},
                }
            },
        }
    )
    scn = _structure_manual(data)
    assert scn.agent_types["walker"].sequences["default"].steps["pursue"].target_agent == 99


def test_block_with_unknown_target_agent_raises() -> None:
    data = _minimal(
        {
            "agents": [{"agent_id": 1, "agent_type": "walker"}],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {"default": {"steps": {"pursue": {"target_agent": 99}}}},
                }
            },
        }
    )
    with pytest.raises(ValueError, match="target_agent=99 does not match"):
        _structure_manual(data)


def test_block_with_target_object_raises() -> None:
    data = _minimal(
        {
            "agents": [{"agent_id": 1, "agent_type": "walker"}, {"agent_id": 99, "agent_type": "walker"}],
            "world_objects": [{"object_id": "b1", "type": "bench", "pose": {"x": 0.0, "y": 0.0, "theta": 0.0}, "capacity": 1}],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {"default": {"steps": {"pursue": {"target_agent": 99, "target_object_type": "bench"}}}},
                }
            },
        }
    )
    with pytest.raises(ValueError, match="target_agent cannot be combined with target_object"):
        _structure_manual(data)


def test_block_with_autonomous_raises() -> None:
    data = _minimal(
        {
            "agents": [{"agent_id": 1, "agent_type": "walker"}, {"agent_id": 99, "agent_type": "walker"}],
            "agent_types": {
                "walker": {
                    "mode": "behavior_tree",
                    "sequences": {"default": {"steps": {"pursue": {"target_agent": 99, "autonomous": True}}}},
                }
            },
        }
    )
    with pytest.raises(ValueError, match="target_agent and autonomous are mutually exclusive"):
        _structure_manual(data)
