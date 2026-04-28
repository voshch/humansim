zone_selection = """
You are an expert in human behavior simulation and spatial reasoning. Your task is to analyze a scenario description and world information to identify relevant semantic zones where agents should operate.

Given:
- User scenario description
- World objects, walls, and obstacles

Identify 3-5 key semantic zones that are most relevant to the scenario. For each zone, provide:
- name: A descriptive name for the zone
- description: What happens in this zone
- relevant_objects: List of world object IDs that belong to this zone
- agent_roles: What types of agents would use this zone

Output as JSON array of zone objects.
"""

spawn_position_selection = """
You are designing spawn positions for agents in a human simulation scenario.

Given:
- User scenario description
- Identified semantic zones with their descriptions
- World layout information

For each agent type mentioned in the scenario, suggest appropriate spawn positions. Consider:
- Natural entry points to the scene
- Logical starting locations based on agent roles
- Avoiding immediate conflicts or unrealistic placements
- Distribution that creates interesting dynamics

Output as JSON with agent_type names as keys, each containing an array of spawn position suggestions with x, y, theta coordinates and reasoning.
"""

behavior_tree_generation = """
You are creating behavior tree configurations for human agents in a simulation. The behaviors should be realistic and based on human psychology and social norms.
BT authoring is split into four fields on the agent type: `needs`, `utility_weights`, `actions`, `sequences` (plus `initial_sequence` to pick the entry sequence; defaults to `"default"`). See [../../arena_humansim/core/behavior/README.md](../../arena_humansim/core/behavior/README.md) for the cross-node invariants (patience phases, seek/cancel semantics, compiler dispatch table).


Given:
- User scenario description
- World semantic zones
- Available agent types (adult, elder)
- World objects and their capabilities

Create agent_type configurations with:
- needs: Psychological or physical needs that drive behavior
- sequences: Behavior sequences with steps, transitions, and timing

Each sequence should have:
- steps: Individual actions with target objects, interactions, durations
- transitions: Conditions that trigger sequence changes
- then: What sequence to go to next

Use realistic timing distributions (mean, std, clip_low, clip_high) and ensure behaviors create natural crowd dynamics.

Output as YAML agent_types section compatible with arena_humansim format.
"""
