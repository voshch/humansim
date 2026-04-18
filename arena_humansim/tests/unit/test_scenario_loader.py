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
