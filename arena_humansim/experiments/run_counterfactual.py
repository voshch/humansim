import os
import sys
import pandas as pd
import subprocess
import yaml

def get_agent_count(scenario_name: str) -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, "scenarios", f"{scenario_name}.yaml")

    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
            
        agents_list = config.get('agents', [])
        
        if not agents_list:
            raise ValueError(f"'agents' list is empty or missing in {scenario_name}.yaml")
            
        return len(agents_list)
        
    except Exception as e:
        print(f"Failed to determine agent count for {scenario_name}.")
        raise e

def build_planner_array(factual_planner: str, target_index: int, total_agents: int, baseline: str = "straight") -> str:
    planners = [factual_planner] * total_agents
    planners[target_index] = baseline
    return ",".join(planners)

def run_counterfactual_sweep(registry_path: str = "/opt/arena_ws/src/arena_humansim/arena_humansim/experiments/data/metrics/failure_registry.csv", 
                             output_dir: str = "/opt/arena_ws/src/arena_humansim/arena_humansim/experiments/data/counterfactual_bags", 
                             baseline_planner: str = "straight"):

    if not os.path.exists(registry_path):
        print(f"Error: Registry not found at {registry_path}. Run Phase 1 first.")
        sys.exit(1)
        
    registry_df = pd.read_csv(registry_path)
    subset_df = registry_df.drop_duplicates(subset=['scenario']).head(5)
    print(f"Loaded {len(subset_df)} failing trials for counterfactual expansion.")

    os.makedirs(output_dir, exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    subset_index = 0
    for index, row in subset_df.iterrows():
        scenario = row['scenario']
        factual_planner = row['ped_planner']
        robot_policy = row['robot_policy']
        seed = row['seed']
        
        total_agents = get_agent_count(scenario)
        
        print(f"\nInitiating Trial [{subset_index+1}/{len(subset_df)}]: {scenario} | Seed: {seed}.")
        print(f"Total Pedestrians (N) = {total_agents}. Launching {total_agents} counterfactual simulations.")

        for i in range(total_agents):
            planner_array = build_planner_array(factual_planner, target_index=i, total_agents=total_agents, baseline=baseline_planner)
            
            bag_name = f"cf_{scenario}_seed{seed}_target{i}"
            bag_path = os.path.join(output_dir, bag_name)

            config = os.path.join(script_dir, "scenarios", f"{scenario}.yaml")
            
            print(f"  -> Running Run {i}/{total_agents-1}: Target Agent {i} set to {baseline_planner}")
            
            cmd = [
                "ros2", "launch", "arena_humansim", "arena_humansim.launch.py",
                f"scenario:={scenario}",
                f"robot_policy:={robot_policy}",
                f"ped_planners:={planner_array}",
                f"seed:={seed}",
                f"record_dir:={bag_path}",
                f"config_path:={config}",

                "record:=true",
                "rtf:=0.0",
                "render:=false",
                "rviz:=false",
                "markers:=0"
            ]
            
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                print(f"Simulation failed for Target Agent {i}. Error: {e}")
                print("Halting sweep due to simulation error.")
                sys.exit(1)

        subset_index += 1

    print("\nCounterfactual sweep complete. All bags saved to:", output_dir)

if __name__ == "__main__":
    run_counterfactual_sweep()
