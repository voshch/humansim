import json
import os
import time
from pathlib import Path
from typing import Any

import attrs
import yaml

from arena_humansim.agents import SampledParams
from arena_humansim.utils.types import AgentState, HighLevelCommand, InteractionState


class SimulationLogger:
    def __init__(self, log_dir: str, seed: int, config: dict[str, Any] | None = None):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._seed = seed
        self._session_start = time.time()

        self._write_config_snapshot(config or {})

        self._log_path = self._log_dir / "session.jsonl"
        self._log_file = open(self._log_path, "w")

    def _write_config_snapshot(self, config: dict[str, Any]):
        snapshot_path = self._log_dir / "config_snapshot.yaml"
        with open(snapshot_path, "w") as f:
            f.write(f"# Config snapshot - arena_humansim\n")
            f.write(f"# Generated at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n\n")
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    def record_agent_spawn(
        self,
        agent_id: int,
        params: SampledParams,
        agent_type_name: str,
    ):
        record = {
            "event": "spawn",
            "agent_id": agent_id,
            "agent_type": agent_type_name,
            "params": attrs.asdict(params),
        }
        self._log_file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def record_agent_despawn(self, agent_id: int, reason: str, tick: int) -> None:
        record = {
            "event": "despawn",
            "tick": tick,
            "agent_id": agent_id,
            "reason": reason,
        }
        self._log_file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def record_tick(
        self,
        tick: int,
        timestamp: float,
        agents: dict[int, AgentState],
        interactions: dict[int, InteractionState],
        commands: dict[int, HighLevelCommand],
    ):
        record = {
            "tick": tick,
            "timestamp": timestamp,
            "agents": {str(aid): _serialize_agent(state) for aid, state in agents.items()},
            "interactions": {str(iid): _serialize_interaction(istate) for iid, istate in interactions.items()},
            "commands": {str(aid): _serialize_command(cmd) for aid, cmd in commands.items()},
        }
        self._log_file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def close(self):
        if self._log_file and not self._log_file.closed:
            self._log_file.flush()
            self._log_file.close()


def _serialize_agent(agent) -> dict[str, Any]:
    return {
        "pose": {"x": agent.pose.x, "y": agent.pose.y, "theta": agent.pose.theta},
        "velocity": {"vx": agent.velocity[0], "vy": agent.velocity[1]},
    }


def _serialize_interaction(interaction) -> dict[str, Any]:
    return {
        "type": int(interaction.type),
        "participants": list(interaction.participants),
        "state": interaction.state,
    }


def _serialize_command(cmd) -> dict[str, Any]:
    return {
        "type": int(cmd.type),
        "target_pose": {
            "x": cmd.target_pose.x,
            "y": cmd.target_pose.y,
            "theta": cmd.target_pose.theta,
        },
        "interaction_target": cmd.interaction_target,
    }
