"""
Behavior Tree Generation Module

This module provides LLM-based generation of behavior tree scripts for human agents
in the arena_humansim simulator. It implements a workflow-based approach using
large language models to create realistic agent behaviors based on scenario descriptions
and world semantic information.
"""

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import yaml
from arena_simulation_setup.tree.World import World, WorldDescription
from cattrs import Converter
from pydantic import TypeAdapter

from arena_humansim.core.behavior.bt_generation.inference_client import GoogleGenAIClient, OpenAIClient
from arena_humansim.core.behavior.bt_generation.schema import AgentConfig, AgentType
from arena_humansim.core.behavior.bt_generation.system_prompts import behavior_tree_generation, spawn_position_selection, zone_selection
from arena_humansim.utils.scenario import WorldObjectConfig
from arena_humansim.utils.types import Pose2D


class GenerationMode(Enum):
    """Generation modes for behavior trees."""

    WORKFLOW = "workflow"
    AGENTIC = "agentic"


@dataclass
class WorldInfo:
    """World semantic information extracted from scenario config."""

    world_description: WorldDescription

    def world_zones_metadata(self) -> str:
        """
        Extract world metadata
        """
        text_parts = []

        text_parts.append("World Zones metadata:")
        for zone in self.world_description.zones:
            corners = [[corner.x, corner.y] for corner in zone.corners]
            objects = [f"{obj.name}" for obj in zone.entities.static]
            objects.extend([f"{door.name}" for door in zone.doors])

            text_parts.append(f"Zone {zone.name} with bounding boxes:\n{corners},\ncontains objects:\n{objects}")

        return "\n".join(text_parts)

    def detail_info(self, zone_names: list[str]) -> str:
        """Convert world info to descriptive text."""
        text_parts = []

        for zone in self.world_description.zones:
            if zone.name in zone_names:
                zone_names.remove(zone.name)
                corners = [[corner.x, corner.y] for corner in zone.corners]
                objects = [f"Object ID: {obj.name}, type: {obj.model.name} at ({obj.pose.position.x}, {obj.pose.position.y})" for obj in zone.entities.static]
                objects.extend([f"{door.name} at start: ({door.start.x}, {door.start.y}), end: ({door.end.x}, {door.end.y})" for door in zone.doors])

                text_parts.append(f"Zone {zone.name} with bounding boxes:\n{corners},\ncontains objects:\n{objects}")

        return "\n".join(text_parts)

    def world_objects(self) -> list[WorldObjectConfig]:
        _world_objects = []
        for zone in self.world_description.zones:
            zone_objects = [WorldObjectConfig(object_id=obj.name, type=obj.model.name, pose=Pose2D(obj.pose.position.x, obj.pose.position.y, obj.pose.orientation.to_yaw())) for obj in zone.entities.static]
            _world_objects.extend(zone_objects)

        return _world_objects


@dataclass
class GenerationContext:
    """Context for generation process."""

    user_prompt: str
    world_info: WorldInfo
    mode: GenerationMode
    existing_agent_types: list[str] | None = None


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

    def generate_scenario(self, context: GenerationContext) -> dict[str, Any]:
        """
        Generate a complete scenario using LLM workflow.

        Args:
            context: Generation context with user prompt and world info

        Returns:
            Complete scenario configuration dictionary
        """
        start = time.time()
        if context.mode == GenerationMode.WORKFLOW:
            scenario = self._generate_workflow(context)
        elif context.mode == GenerationMode.AGENTIC:
            scenario = self._generate_agentic(context)
        else:
            raise ValueError(f"Unsupported generation mode: {context.mode}")
        end = time.time()
        print(f"Scenario generation took: {(end - start):.2f}s")

        return scenario

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

    def _select_zones(self, context: GenerationContext) -> list[str]:
        """Stage 1: Select relevant semantic zones."""
        system_prompt = zone_selection
        user_prompt = f"Scenario: {context.user_prompt}\nWorld Information:\n{context.world_info.world_zones_metadata()}\nIdentify the most relevant semantic zones for this scenario."

        type_adapter = TypeAdapter(list[str])

        start = time.time()
        response = self.llm_client.generate(contents=user_prompt, system_instruction=system_prompt, response_json_schema=type_adapter.json_schema())
        end = time.time()

        assert isinstance(response, str)
        zones = type_adapter.validate_json(response)

        print(f"Zones selection took: {(end - start):.2f}s")

        return zones

    def _select_spawn_positions(self, context: GenerationContext, zone_names: list[str]) -> list[AgentConfig]:
        """Stage 2: Select spawn positions for agents."""
        system_prompt = spawn_position_selection
        user_prompt = f"Scenario: {context.user_prompt}\nWorld Information:\n{context.world_info.detail_info(zone_names)}\nSuggest spawn positions for different agent types."

        type_adaptor = TypeAdapter(list[AgentConfig])

        start = time.time()
        response = self.llm_client.generate(system_prompt, user_prompt, response_json_schema=type_adaptor.json_schema())
        end = time.time()

        assert isinstance(response, str)
        agent_configs = type_adaptor.validate_json(response)

        print(f"Spawn positions selection took: {(end - start):.2f}s")

        return agent_configs

    def _generate_behavior_trees(self, context: GenerationContext, zone_names: list[str], spawn_positions: list[AgentConfig]) -> dict[str, AgentType]:
        """Stage 3: Generate behavior trees for agent types."""
        system_prompt = behavior_tree_generation
        _spawn_positions = [s.model_dump() for s in spawn_positions]
        _world_info = context.world_info.detail_info(zone_names)
        user_prompt = f"Scenario: {context.user_prompt}\nWorld Information:\n{_world_info}\nAgent Types and Spawn Positions:\n{_spawn_positions}\nCreate behavior tree configurations for the agent types in this scenario."

        type_adapter = TypeAdapter(dict[str, AgentType])

        start = time.time()
        response = self.llm_client.generate(system_prompt, user_prompt, response_json_schema=type_adapter.json_schema())
        end = time.time()

        assert isinstance(response, str)
        agent_types = type_adapter.validate_json(response)

        print(f"Behavior generation took: {(end - start):.2f}s")

        return agent_types

    def _assemble_scenario(self, context: GenerationContext, agent_types: dict[str, AgentType], spawn_positions: list[AgentConfig]) -> dict[str, Any]:
        """Stage 4: Assemble complete scenario configuration."""
        converter = Converter()
        scenario = {
            "name": f"generated_{context.user_prompt.replace(' ', '_')[:50]}",
            "description": context.user_prompt,
            "simulation": {"seed": 42, "dt": 0.05, "bt_tick_interval": 5, "execution_mode": "master"},
            "modules": {"perception": "default", "global_planner": "dijkstra", "local_planner": "sfm", "animation": "kinematic"},
            "agent_types": {k: v.model_dump() for k, v in agent_types.items()},
            "agents": [s.model_dump() for s in spawn_positions],
            "world_objects": converter.unstructure(world_info.world_objects()),
        }

        return scenario


# Example usage
if __name__ == "__main__":
    from pathlib import Path

    from ament_index_python.packages import get_package_share_directory

    world_path = Path(os.path.join(get_package_share_directory("arena_simulation_setup"), "worlds", "hospital_1"))
    arena_world = World(path=world_path)

    world_info = WorldInfo(world_description=arena_world.load())

    # Create generator with mock LLM
    generator = LLMBehaviorTreeGenerator()

    # Generate scenario
    context = GenerationContext(user_prompt="A busy office where workers perform tasks and take breaks for food and water", world_info=world_info, mode=GenerationMode.WORKFLOW)

    start = time.time()
    scenario_yaml = generator.generate_yaml(context)
    end = time.time()
    print(f"Generation time: {(end - start):.2f}s")
    print(scenario_yaml)
    scenario_path = os.path.join(get_package_share_directory("arena_humansim"), "config", "scenarios", "test_scenario_generator.yaml")
    generator.save_scenario(scenario_path)
