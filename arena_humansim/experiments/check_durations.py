import os
import yaml

bags_dir = "/opt/arena_ws/src/arena_humansim/arena_humansim/experiments/data/counterfactual_bags"

bag_dirs = sorted([
    d for d in os.listdir(bags_dir)
    if d.startswith("cf_") and os.path.isdir(os.path.join(bags_dir, d))
])

for bag_name in bag_dirs:
    bag_path = os.path.join(bags_dir, bag_name)
    metadata_path = os.path.join(bag_path, "bag", "metadata.yaml")
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(bag_path, "metadata.yaml")

    with open(metadata_path) as f:
        meta = yaml.safe_load(f)

    duration_ns = meta["rosbag2_bagfile_information"]["duration"]["nanoseconds"]
    duration_s = duration_ns / 1e9
    print(f"{duration_s:7.2f}s  {bag_name}")
