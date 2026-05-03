import argparse
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description="Automate arena_humansim ablations.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        required=True,
        help="List of scenarios to run (e.g., --scenarios queue bottleneck)",
    )
    parser.add_argument(
        "--planners",
        nargs="+",
        required=True,
        help="List of local planners to test (e.g., --planners sfm socialgail)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="List of random seeds to test (e.g., --seeds 42 100 999)",
    )
    parser.add_argument(
        "--sim_duration",
        type=int,
        required=True,
        help="Exact simulated seconds to record",
    )

    args = parser.parse_args()

    scenarios = args.scenarios
    planners = args.planners
    seeds = args.seeds
    sim_duration = args.sim_duration

    total_runs = len(scenarios) * len(planners) *  len(seeds)
    current_run = 0

    for scenario in scenarios:
        for planner in planners:
            for seed in seeds:

                current_run += 1
                print("=========================================")
                print(f"Running ablation ({current_run}/{total_runs}): Scenario={scenario}, Planner={planner}, Seed={seed}")
                print("=========================================")

                cmd = [
                    "ros2",
                    "launch",
                    "arena_humansim",
                    "arena_humansim.launch.py",
                    f"scenario:={scenario}",
                    f"local_planner:={planner}",
                    f"seed:={seed}",
                    "rtf:=0",
                    f"time:={sim_duration}",
                    "record:=True",
                    "render:=False",
                    "rviz:=false",
                    "markers:=0",
                ]

                try:
                    print(f"Simulation started at max speed. Waiting for {args.sim_duration} sim-seconds to elapse...")

                    subprocess.run(cmd, check=True)

                    print("Simulation closed cleanly.")

                    if current_run < total_runs:
                        print("Waiting 5 seconds to clear ROS network before next run...")
                        time.sleep(5)

                except subprocess.CalledProcessError as e:
                    print(f"An error occurred during {scenario} with {planner} (Seed {seed}): {e}")

    print("All ablations completed!")


if __name__ == "__main__":
    main()
