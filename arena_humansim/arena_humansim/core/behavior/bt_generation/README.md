# Behavior Trees Generation Pipeline

## Overview

This pipeline provides **two approaches** for LLM-based generation of behavior tree scripts for human agents in the arena_humansim simulator:

1. **WORKFLOW Mode**: Fast, predefined stages with simple flow
2. **AGENTIC Mode**: Iterative refinement with validation feedback using LangGraph

### Inputs
- **Scenario description** (User prompt): Natural language description of the scenario
- **World description**: Automatically parsed from `ScenarioConfig.world_objects`, `ScenarioConfig.walls`, and `ScenarioConfig.obstacles`
- **Generation mode**: User selects WORKFLOW or AGENTIC

### Outputs
- Generated scenario in `scenario.yaml` format with `agent_types` and `agents` fields
- Example: `arena_humansim/config/scenarios/queue_demo.yaml`

## Architecture

The generation system uses a multi-stage LLM workflow:

1. **Zone Selection**: LLM analyzes scenario and world info to identify semantic zones
2. **Spawn Position Selection**: LLM suggests natural spawn positions based on zones
3. **Behavior Tree Generation**: LLM creates agent behaviors with needs, sequences, and transitions
4. **Scenario Assembly**: System combines all components into valid YAML

### WORKFLOW Mode Flow
```
[Zone Selection] -> [Spawn Positions] -> [Behavior Trees] -> [Assembly]
```
- Single pass through each stage
- Fast and deterministic
- No feedback or refinement

### AGENTIC Mode Flow (LangGraph)
T.B.D

## Type Enforcement

- `TypeValidator` ensures generated YAML conforms to arena_humansim schema
- Automatic YAML repair for common formatting issues
- Validation of agent type configurations and sequences