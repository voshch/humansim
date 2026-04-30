import subprocess
import time
import os
import signal

scenarios = ["queue", "corridor" , "bottleneck"]
planners = ["sfm", "socialgail"]

RUN_DURATION = 60

total_runs = len(scenarios) * len(planners)
current_run = 0

for scenario in scenarios:
    for planner in planners:
        current_run += 1
        print(f"=========================================")
        print(f"Running ablation ({current_run}/{total_runs}): Scenario={scenario}, Planner={planner}")
        print(f"=========================================")
        
        cmd = [
            "ros2", "launch", "arena_humansim", "arena_humansim.launch.py",
            f"scenario:={scenario}",
            f"local_planner:={planner}",
            "record:=True",
            "render:=False",
            "rviz:=false",
            "markers:=0"
        ]
        
        try:
            process = subprocess.Popen(cmd, preexec_fn=os.setsid)
            
            print(f"Simulation started. Waiting for {RUN_DURATION} seconds...")
            time.sleep(RUN_DURATION)
            
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