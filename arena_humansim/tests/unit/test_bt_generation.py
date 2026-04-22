"""Tests for behavior tree generation."""

import pytest
import yaml
from arena_humansim.core.behavior.bt_generation import (
    LLMBehaviorTreeGenerator,
    GenerationContext,
    GenerationMode,
    WorldInfo,
    TypeValidator,
)


class TestLLMBehaviorTreeGenerator:
    """Test the LLM-based behavior tree generator."""

    def test_workflow_generation(self):
        """Test WORKFLOW mode generation."""
        world_info = WorldInfo(
            objects=[
                {"object_id": "workstation", "type": "workstation", "pose": {"x": 0, "y": 0, "theta": 0}},
                {"object_id": "cafeteria", "type": "cafeteria", "pose": {"x": 5, "y": 0, "theta": 0}}
            ],
            walls=[],
            obstacles=[],
            zones=[]
        )

        generator = LLMBehaviorTreeGenerator()
        context = GenerationContext(
            user_prompt="Office workers scenario",
            world_info=world_info,
            mode=GenerationMode.WORKFLOW
        )

        scenario = generator.generate_scenario(context)

        assert "name" in scenario
        assert "simulation" in scenario
        assert "modules" in scenario
        assert "agent_types" in scenario
        assert "agents" in scenario

    def test_yaml_generation(self):
        """Test YAML string generation."""
        world_info = WorldInfo(
            objects=[{"object_id": "test", "type": "workstation", "pose": {"x": 0, "y": 0, "theta": 0}}],
            walls=[],
            obstacles=[],
            zones=[]
        )

        generator = LLMBehaviorTreeGenerator()
        context = GenerationContext(
            user_prompt="Test scenario",
            world_info=world_info,
            mode=GenerationMode.WORKFLOW
        )

        yaml_str = generator.generate_yaml(context)

        # Should be valid YAML
        data = yaml.safe_load(yaml_str)
        assert isinstance(data, dict)
        assert "name" in data

    def test_type_validator(self):
        """Test type validation."""
        # Valid agent type config
        valid_config = {
            "extends": "adult",
            "mode": "behavior_tree",
            "sequences": {
                "test_seq": {
                    "steps": {
                        "test_step": {
                            "target_object_type": "workstation",
                            "interaction": "WORK"
                        }
                    }
                }
            }
        }

        assert TypeValidator.validate_agent_type_config(valid_config)

        # Invalid config (missing extends)
        invalid_config = {
            "mode": "behavior_tree",
            "sequences": {}
        }

        assert not TypeValidator.validate_agent_type_config(invalid_config)

    def test_yaml_repair(self):
        """Test YAML repair functionality."""
        # Valid YAML
        valid_yaml = "name: test\nextends: adult\n"
        repaired = TypeValidator.repair_yaml(valid_yaml)
        assert "name: test" in repaired

        # Invalid YAML (should still return something)
        invalid_yaml = "invalid: yaml: content: [unclosed"
        repaired = TypeValidator.repair_yaml(invalid_yaml)
        assert isinstance(repaired, str)


class TestWorldInfo:
    """Test WorldInfo dataclass."""

    def test_world_info_creation(self):
        """Test WorldInfo creation and text conversion."""
        world_info = WorldInfo(
            objects=[
                {"object_id": "workstation", "type": "workstation", "pose": {"x": 0, "y": 0, "theta": 0}}
            ],
            walls=[],
            obstacles=[],
            zones=[
                {"name": "work_zone", "description": "Work area", "relevant_objects": ["workstation"], "agent_roles": ["worker"]}
            ]
        )

        text = world_info.to_text()
        assert "workstation" in text
        assert "work_zone" in text