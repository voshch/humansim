import warnings
from pathlib import Path

import pandas as pd
from rosbags.highlevel import AnyReader

warnings.filterwarnings("ignore", category=RuntimeWarning)


def extract_agent_states(bag_path: Path) -> pd.DataFrame:
    extracted_data = []
    with AnyReader([bag_path]) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if "agent_states" not in connection.topic:
                continue
            msg = reader.deserialize(rawdata, connection.msgtype)
            time_sec = timestamp * 1e-9
            for agent in msg.agents:
                extracted_data.append(
                    {
                        "time": time_sec,
                        "agent_id": agent.agent_id,
                        "x": agent.pose.x,
                        "y": agent.pose.y,
                        "vx": agent.velocity.x,
                        "vy": agent.velocity.y,
                        "radius": agent.radius,
                        "planner": agent.policy,
                    }
                )
    return pd.DataFrame(extracted_data)
