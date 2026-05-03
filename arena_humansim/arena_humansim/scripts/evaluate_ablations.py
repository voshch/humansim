import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader

warnings.filterwarnings("ignore", category=RuntimeWarning)

def extract_agent_states(bag_path):
    extracted_data = []
    with AnyReader([bag_path]) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if 'agent_states' not in connection.topic:
                continue
            msg = reader.deserialize(rawdata, connection.msgtype)
            time_sec = timestamp * 1e-9 
            for agent in msg.agents:
                extracted_data.append({
                    'time': time_sec,
                    'agent_id': agent.agent_id,
                    'x': agent.pose.x,
                    'y': agent.pose.y,
                    'vx': agent.velocity.x,
                    'vy': agent.velocity.y,
                    'radius': agent.radius,
                    'planner': agent.policy
                })
    return pd.DataFrame(extracted_data)

def calculate_kinematic_metrics(agent_df):
    dt = 0.05 
    v = agent_df[['vx', 'vy']].values
    v_next = np.roll(v, -1, axis=0)
    v_prev = np.roll(v, 1, axis=0)
    jerk_vec = (v_next - 2 * v + v_prev) / (dt ** 2)
    jerk_vec = jerk_vec[1:-1]
    jerk_mag = np.linalg.norm(jerk_vec, axis=1)
    mean_jerk = np.nanmean(jerk_mag)
    vx = agent_df['vx'].values
    vy = agent_df['vy'].values
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    numerator = np.abs(vx * ay - vy * ax)
    denominator = (vx**2 + vy**2)**1.5
    curvature = np.where(denominator > 1e-5, numerator / denominator, 0)
    mean_curvature = np.nanmean(curvature)
    return pd.Series({'mean_jerk': mean_jerk, 'mean_curvature': mean_curvature})

def calculate_run_collisions(run_df):
    collision_pairs = set()
    for _, frame in run_df.groupby('time'):
        if len(frame) < 2:
            continue
        agent_ids = frame['agent_id'].values
        coords = frame[['x', 'y']].values
        radii = frame['radius'].values
        dx = coords[:, 0:1] - coords[:, 0]
        dy = coords[:, 1:2] - coords[:, 1]
        distances = np.sqrt(dx**2 + dy**2)
        thresholds = radii[:, None] + radii
        collision_mask = distances < thresholds
        np.fill_diagonal(collision_mask, False)
        colliding_indices = np.argwhere(collision_mask)
        for i, j in colliding_indices:
            pair = frozenset([agent_ids[i], agent_ids[j]])
            collision_pairs.add(pair)
    return len(collision_pairs)

def main():
    parser = argparse.ArgumentParser(description="Evaluate arena_humansim ablations.")
    parser.add_argument("--recordings_dir", type=str, required=True)
    args = parser.parse_args()
    recordings_path = Path(args.recordings_dir)
    all_metrics = []
    print(f"Scanning {recordings_path} for ablation data...\n")
    for run_dir in recordings_path.iterdir():
        if run_dir.is_dir():
            bag_dir = run_dir / "bag"
            if bag_dir.exists():
                folder_name = run_dir.name
                scenario_name = folder_name.split('_')[-1] if '_' in folder_name else "unknown"
                df = extract_agent_states(bag_dir)
                if df.empty:
                    continue
                planner_name = df['planner'].iloc[0] if 'planner' in df.columns else "unknown"
                total_collisions = calculate_run_collisions(df)
                df = df.sort_values(['agent_id', 'time'])
                run_metrics = df.groupby('agent_id').apply(calculate_kinematic_metrics).reset_index()
                avg_run_jerk = run_metrics['mean_jerk'].mean()
                avg_run_curvature = run_metrics['mean_curvature'].mean()
                all_metrics.append({
                    'Scenario': scenario_name,
                    'Planner': planner_name,
                    'Jerk': avg_run_jerk,
                    'Curvature': avg_run_curvature,
                    'Collisions': total_collisions
                })
    results_df = pd.DataFrame(all_metrics)
    if not results_df.empty:
        summary_df = results_df.groupby(['Scenario', 'Planner']).mean().reset_index()
        print("\n=======================================================")
        print("          FINAL ABLATION METRICS SUMMARY                 ")
        print("=======================================================")
        print(summary_df.to_string(index=False, float_format="{:.4f}".format))
        print("=======================================================\n")
    else:
        print("No valid metrics were generated. Check bag data.")

if __name__ == "__main__":
    main()
