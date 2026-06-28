import pandas as pd
import os
import sys

def build_failure_registry(metrics_dir: str = "/data/metrics", 
                           output_file: str = "/opt/arena_ws/src/arena_humansim/arena_humansim/experiments/data/metrics/failure_registry.csv"):
    robot_metrics_path = os.path.join(metrics_dir, "robot_metrics.parquet")
    failures_path = os.path.join(metrics_dir, "failures.parquet")

    if not os.path.exists(robot_metrics_path) or not os.path.exists(failures_path):
        print(f"Error: Required dataset files not found in {metrics_dir}.")
        print("Ensure 'robot_metrics.parquet' and 'failures.parquet' are downloaded.")
        sys.exit(1)

    print("Loading PEDS-37 datasets.")
    
    robot_metrics_df = pd.read_parquet(robot_metrics_path)
    failures_df = pd.read_parquet(failures_path)

    print(f"Total recorded robot trials: {len(robot_metrics_df)}")

    failed_runs_df = robot_metrics_df[robot_metrics_df['success'] == 0]
    print(f"Isolated {len(failed_runs_df)} failing trials.")

    merge_keys = [
        'scenario', 
        'bucket', 
        'ped_planner', 
        'robot_policy', 
        'seed', 
        'source_dir'
    ]

    merged_df = pd.merge(
        failed_runs_df,
        failures_df,
        on=merge_keys,
        how='left'
    )

    target_columns = [
        'scenario', 
        'bucket', 
        'ped_planner', 
        'robot_policy', 
        'seed', 
        'cause', 
        'time_to_goal_s', 
        'n_robot_collisions',
        'time_to_goal_s', 
        'ttg_ratio',
        'path_efficiency',
        'n_robot_collisions',
        'personal_space_violations',
        'psv_per_sec',
        'frozen_at_end',
        'final_goal_dist'
    ]
    
    registry_df = merged_df[target_columns].drop_duplicates()

    registry_df.to_csv(output_file, index=False)
    print(f"\nSuccess! Failure registry saved to: {output_file}")

if __name__ == "__main__":
    build_failure_registry()
