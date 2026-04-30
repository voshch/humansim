import argparse
import os
import signal
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
        "--duration",
        type=int,
        required=True,
        help="Duration of each simulation run in seconds",
    )

    args = parser.parse_args()

    scenarios = args.scenarios
    planners = args.planners
    run_duration = args.duration

    total_runs = len(scenarios) * len(planners)
    current_run = 0

    for scenario in scenarios:
        for planner in planners:
            current_run += 1
            print("=========================================")
            print(f"Running ablation ({current_run}/{total_runs}): Scenario={scenario}, Planner={planner}")
            print("=========================================")

            cmd = [
                "ros2",
                "launch",
                "arena_humansim",
                "arena_humansim.launch.py",
                f"scenario:={scenario}",
                f"local_planner:={planner}",
                "record:=True",
                "render:=False",
                "rviz:=false",
                "markers:=0",
            ]

            try:
                process = subprocess.Popen(cmd, preexec_fn=os.setsid)

                print(f"Simulation started. Waiting for {run_duration} seconds...")
                time.sleep(run_duration)

                print("Sending SIGINT to terminate simulation and save bag...")
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
                process.wait()

                print("Simulation closed cleanly.")

                if current_run < total_runs:
                    print("Waiting 5 seconds before next run...")
                    time.sleep(5)

            except Exception as e:
                print(f"An error occurred during {scenario} with {planner}: {e}")

    print("All ablations completed!")


if __name__ == "__main__":
    main()
    