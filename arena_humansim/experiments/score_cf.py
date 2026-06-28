import os
import sys
import yaml
import pandas as pd

def get_bag_duration_ns(bag_dir: str) -> int:
    metadata_path = os.path.join(bag_dir, "bag", "metadata.yaml")
    
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(bag_dir, "metadata.yaml")
        
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.yaml not found in expected layouts for {bag_dir}")
        
    with open(metadata_path, "r") as f:
        meta = yaml.safe_load(f)
        
    return meta["rosbag2_bagfile_information"]["duration"]["nanoseconds"]

KNOWN_TIMEOUTS_S = {67.2, 162.7, 199.9, 299.9, 399.9}
TOLERANCE_S = 0.5

def check_if_success(duration_ns: int) -> bool:
    duration_s = round(duration_ns / 1e9, 1)
    for timeout in KNOWN_TIMEOUTS_S:
        if abs(duration_s - timeout) < TOLERANCE_S:
            return False
    return True

def parse_bag_name(bag_name: str) -> dict:
    try:
        rest, target_str = bag_name.rsplit("_target", 1)
        rest, planner_str = rest.rsplit("_planner", 1)
        scenario_part, seed_str = rest.rsplit("_seed", 1)
        scenario = scenario_part.replace("cf_", "", 1)
        return {
            "scenario": scenario,
            "seed": int(seed_str),
            "ped_planner": planner_str,
            "target": int(target_str),
        }
    except ValueError:
        raise ValueError(f"Could not parse bag name: {bag_name}")

def score_counterfactuals(
    bags_dir: str = "/opt/arena_ws/src/arena_humansim/arena_humansim/experiments/data/counterfactual_bags",
    registry_path: str = "/opt/arena_ws/src/arena_humansim/arena_humansim/experiments/data/metrics/failure_registry.csv",
    output_path: str = "/opt/arena_ws/src/arena_humansim/arena_humansim/experiments/data/metrics/causal_scores.csv",
):
    if not os.path.exists(bags_dir):
        print(f"Error: Bags target directory not found at {bags_dir}")
        sys.exit(1)
        
    if not os.path.exists(registry_path):
        print(f"Error: Failure registry baseline data not found at {registry_path}. Cannot calculate Deltas.")
        sys.exit(1)

    registry_df = pd.read_csv(registry_path)
    registry_df = registry_df[registry_df['robot_policy'] != 'drlvo']

    bag_dirs = sorted([
        d for d in os.listdir(bags_dir)
        if d.startswith("cf_") and os.path.isdir(os.path.join(bags_dir, d))
    ])

    if not bag_dirs:
        print(f"No counterfactual execution bags matched inside {bags_dir}")
        sys.exit(1)

    print(f"Found {len(bag_dirs)} counterfactual configurations. Scoring profiles...")

    rows = []
    missing = []

    for bag_name in bag_dirs:
        bag_path = os.path.join(bags_dir, bag_name)
        try:
            info = parse_bag_name(bag_name)
            duration_ns = get_bag_duration_ns(bag_path)
            
            succeeded = check_if_success(duration_ns)

            rows.append({
                "scenario":    info["scenario"],
                "seed":        info["seed"],
                "ped_planner": info["ped_planner"],
                "target":      info["target"],
                "duration_s":  round(duration_ns / 1e9, 2),
                "succeeded":   succeeded,
                "causal":      succeeded, 
            })

        except FileNotFoundError:
            missing.append(bag_name)
        except ValueError as e:
            print(f"  Warning: {e}")

    if missing:
        print(f"\nWarning: metadata.yaml missing for {len(missing)} bag(s):")
        for m in missing:
            print(f"  {m}")

    if not rows:
        print("Error: No bags scored successfully. Check folder structure.")
        sys.exit(1)

    results_df = pd.DataFrame(rows)

    merged_df = pd.merge(
    results_df,
    registry_df[['scenario', 'seed', 'ped_planner', 'robot_policy', 'cause']],
    on=['scenario', 'seed', 'ped_planner'],
    how='left'
    )

    merged_df = merged_df.sort_values(["scenario", "seed", "target"]).reset_index(drop=True)

    print("\n=== Causal Attribution Matrix Summary ===\n")
    for (scenario, seed), group in merged_df.groupby(["scenario", "seed"]):
        planner = group["ped_planner"].iloc[0] if not pd.isna(group["ped_planner"].iloc[0]) else "Unknown"
        policy = group["robot_policy"].iloc[0] if not pd.isna(group["robot_policy"].iloc[0]) else "Unknown"
        original_cause = group["cause"].iloc[0] if not pd.isna(group["cause"].iloc[0]) else "Unknown"
        
        causal_targets = group[group["causal"] == True]["target"].tolist()
        total_agents = len(group)
        
        print(f"Trial: {scenario} | Seed: {seed} | Planner: {planner} | Policy: {policy}")
        print(f"  -> Factual Failure: {original_cause}")
        print(f"  -> Causal Influence: {len(causal_targets)}/{total_agents} agents responsible. IDs: {causal_targets}\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"Scores saved to: {output_path}")

if __name__ == "__main__":
    score_counterfactuals()
