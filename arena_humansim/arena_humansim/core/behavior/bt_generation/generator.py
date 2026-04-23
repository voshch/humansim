"""
Behavior Tree Generation Module

This module provides LLM-based generation of behavior tree scripts for human agents
in the arena_humansim simulator. It implements a workflow-based approach using
large language models to create realistic agent behaviors based on scenario descriptions
and world semantic information.
"""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import attrs
import cattrs
import yaml
from apischema import settings
from apischema.json_schema import deserialization_schema
from apischema.objects import ObjectField

from arena_humansim.core.agents.types import AgentType
from arena_humansim.core.behavior.bt_generation.inference_client import GoogleGenAIClient, OpenAIClient
from arena_humansim.core.behavior.bt_generation.system_prompts import behavior_tree_generation, spawn_position_selection, zone_selection
from arena_humansim.utils.scenario import ScenarioConfig

prev_default_object_fields = settings.default_object_fields


def attrs_fields(cls: type) -> Sequence[ObjectField] | None:
    if hasattr(cls, "__attrs_attrs__"):
        return [ObjectField(a.name, a.type, required=a.default == attrs.NOTHING, default=a.default) for a in cls.__attrs_attrs__]

    else:
        return prev_default_object_fields(cls)


settings.default_object_fields = attrs_fields


class GenerationMode(Enum):
    """Generation modes for behavior trees."""

    WORKFLOW = "workflow"
    AGENTIC = "agentic"


@dataclass
class WorldInfo:
    """World semantic information extracted from scenario config."""

    scenario_config: ScenarioConfig

    def to_text(self) -> str:
        """Convert world info to descriptive text."""
        text_parts = []

        text_parts.append("World Objects:")
        for obj in self.scenario_config.world_objects:
            text_parts.append(f"- Object {obj.type} {obj.object_id} at position ({obj.pose.x}, {obj.pose.y})")

        return "\n".join(text_parts)


@dataclass
class GenerationContext:
    """Context for generation process."""

    user_prompt: str
    world_info: WorldInfo
    mode: GenerationMode
    existing_agent_types: list[str] | None = None


class TypeValidator:
    """Typed validation + parsing using cattrs."""

    def __init__(self):
        self.converter = cattrs.Converter()

    @staticmethod
    def normalize_yaml_format(yaml_str: str) -> str:
        try:
            data = yaml.safe_load(yaml_str)
            return yaml.dump(data, default_flow_style=False, sort_keys=False)
        except yaml.YAMLError:
            return yaml_str

    def parse_agent_types(self, yaml_str: str | None) -> dict[str, AgentType]:
        """
        Parse and validate agent_types from YAML string.

        Returns:
            dict[str, AgentType]

        Raises:
            RuntimeError if parsing or structuring fails
        """
        assert yaml_str is not None

        # normalize (optional)
        yaml_str = self.normalize_yaml_format(yaml_str)

        # YAML to dict
        try:
            raw = yaml.safe_load(yaml_str)
        except yaml.YAMLError as e:
            print("Raw yaml string:\n", yaml_str)
            raise RuntimeError("yaml error") from e

        if not isinstance(raw, dict):
            raise RuntimeError("Agent types YAML must be a dict")

        # dict to typed
        try:
            return {name: self.converter.structure(config, AgentType) for name, config in raw.items()}
        except Exception as e:
            raise RuntimeError("structure error") from e


class LLMBehaviorTreeGenerator:
    """LLM-based generator for behavior tree configurations."""

    def __init__(self):
        _endpoint = os.environ.get("LLM_API_ENDPOINT")
        if not _endpoint:
            self.llm_client = GoogleGenAIClient(model="gemini-3-flash-preview")
        elif _endpoint == "GOOGLEGENAI":
            self.llm_client = GoogleGenAIClient(model="gemini-3-flash-preview")
        elif _endpoint == "OPENAI":
            self.llm_client = OpenAIClient(model="...")
        else:
            raise ValueError(f"LLM_API_ENDPOINT must be one of ['GOOGLEGENAI', 'OPENAI'], got {_endpoint}")

        self.generated_scenario: str | None = None
        # Kinda useless for now
        # I tend to use `TypeValidator` for OpenAI API if I couldn't find a way to use it with apischema
        self.type_validator = TypeValidator()

    def generate_scenario(self, context: GenerationContext) -> dict[str, Any]:
        """
        Generate a complete scenario using LLM workflow.

        Args:
            context: Generation context with user prompt and world info

        Returns:
            Complete scenario configuration dictionary
        """
        if context.mode == GenerationMode.WORKFLOW:
            return self._generate_workflow(context)
        elif context.mode == GenerationMode.AGENTIC:
            return self._generate_agentic(context)
        else:
            raise ValueError(f"Unsupported generation mode: {context.mode}")

    def generate_yaml(self, context: GenerationContext) -> str:
        """Generate scenario as YAML string."""
        scenario = self.generate_scenario(context)
        self.generated_scenario = yaml.dump(scenario, default_flow_style=False, sort_keys=False)

        return yaml.dump(scenario, default_flow_style=False, sort_keys=False)

    def save_scenario(self, filepath: str):
        """Save scenario to file."""
        assert self.generated_scenario is not None, "Scenario was not generated"

        try:
            with open(filepath, 'w') as f:
                f.write(self.generated_scenario)
            print(f"Saved scenario at {filepath}")
        except Exception as e:
            print(f"Error occured while trying to save scenario: {e}")

    def _generate_workflow(self, context: GenerationContext) -> dict[str, Any]:
        """Generate using predefined workflow stages."""
        if not self.llm_client:
            raise ValueError("LLM client required for WORKFLOW mode")

        # Stage 1: Zone selection
        zones = self._select_zones(context)

        # Stage 2: Spawn position selection
        spawn_positions = self._select_spawn_positions(context, zones)

        # Stage 3: Behavior tree generation
        agent_types = self._generate_behavior_trees(context, zones, spawn_positions)

        # Stage 4: Assemble scenario
        return self._assemble_scenario(context, agent_types, spawn_positions)

    def _generate_agentic(self, context: GenerationContext) -> dict[str, Any]:
        """Generate using agentic AI approach with LangGraph."""
        raise NotImplementedError()

    def _select_zones(self, context: GenerationContext) -> list[dict[str, Any]]:
        """Stage 1: Select relevant semantic zones."""
        system_prompt = zone_selection
        user_prompt = f"Scenario: {context.user_prompt}\nWorld Information:\n{context.world_info.to_text()}\nIdentify the most relevant semantic zones for this scenario."

        response = self.llm_client.generate(contents=user_prompt, system_instruction=system_prompt)
        assert isinstance(response, str)

        try:
            zones = json.loads(response)
            return zones if isinstance(zones, list) else []
        except json.JSONDecodeError:
            return []

    def _select_spawn_positions(self, context: GenerationContext, zones: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """Stage 2: Select spawn positions for agents."""
        system_prompt = spawn_position_selection
        user_prompt = f"Scenario: {context.user_prompt}\nSemantic Zones:\n{json.dumps(zones, indent=2)}\nWorld Information:\n{context.world_info.to_text()}\nSuggest spawn positions for different agent types."

        response = self.llm_client.generate(system_prompt, user_prompt)
        assert isinstance(response, str)

        try:
            positions = json.loads(response)
            return positions if isinstance(positions, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _generate_behavior_trees(self, context: GenerationContext, zones: list[dict[str, Any]], spawn_positions: dict[str, list[dict[str, Any]]]) -> dict[str, AgentType]:
        """Stage 3: Generate behavior trees for agent types."""
        system_prompt = behavior_tree_generation
        user_prompt = f"Scenario: {context.user_prompt}\nSemantic Zones:\n{json.dumps(zones, indent=2)}\nAgent Types and Spawn Positions:\n{json.dumps(spawn_positions, indent=2)}\nWorld Information:\n{context.world_info.to_text()}\nCreate behavior tree configurations for the agent types in this scenario."

        response = self.llm_client.generate(system_prompt, user_prompt, response_json_schema=deserialization_schema(dict[str, AgentType]))
        assert isinstance(response, dict), f"Parsing error. Got response from LLM: {response}"

        for agent_type_name, agent_type in response.items():
            if not isinstance(agent_type, AgentType) or not isinstance(agent_type_name, str):
                raise ValueError(f"Parsing error of agent {agent_type_name}. Got response from LLM: {response}")

        return response

    def _assemble_scenario(self, context: GenerationContext, agent_types: dict[str, AgentType], spawn_positions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        """Stage 4: Assemble complete scenario configuration."""
        scenario = {
            "name": f"generated_{context.user_prompt.replace(' ', '_')[:50]}",
            "description": context.user_prompt,
            "simulation": {"seed": 42, "dt": 0.05, "bt_tick_interval": 5, "execution_mode": "master"},
            "modules": {"perception": "default", "global_planner": "dijkstra", "local_planner": "sfm", "animation": "kinematic"},
            "world_objects": context.world_info.scenario_config.world_objects,
            "agent_types": agent_types,
            "agents": [],
        }

        # Create agent instances from spawn positions
        agent_id = 1
        for agent_type, positions in spawn_positions.items():
            for pos in positions:
                scenario["agents"].append({"agent_id": agent_id, "agent_type": agent_type, "spawn_pose": {"x": pos.get("x", 0), "y": pos.get("y", 0), "theta": pos.get("theta", 0)}})
                agent_id += 1

        return scenario


# Example usage
if __name__ == "__main__":
    from arena_humansim.utils.scenario import WorldObjectConfig
    from arena_humansim.utils.types import Pose2D

    world_info = WorldInfo(
        ScenarioConfig(
            world_objects=[WorldObjectConfig(object_id="workstation", type="workstation", pose=Pose2D(0.0, 0.0, 0.0)), WorldObjectConfig(object_id="cafeteria", type="cafeteria", pose=Pose2D(5.0, 0.0, 0.0)), WorldObjectConfig(object_id="water_fountain", type="water_fountain", pose=Pose2D(3.0, 2.0, 0.0))],
        )
    )

    # Create generator with mock LLM
    generator = LLMBehaviorTreeGenerator()

    # Generate scenario
    context = GenerationContext(user_prompt="A busy office where workers perform tasks and take breaks for food and water", world_info=world_info, mode=GenerationMode.WORKFLOW)

    scenario_yaml = generator.generate_yaml(context)
    print(scenario_yaml)
    generator.save_scenario("/opt/arena/deps/humansim/arena_humansim/arena_humansim/config/scenarios/test_scenario_generator.yaml")
