# State of a single pedestrian agent
int32 agent_id
geometry_msgs/Pose2D pose
geometry_msgs/Vector3 velocity       # linear velocity (x, y, z=0)
float64 desired_velocity              # preferred walking speed (m/s)
float64 radius                        # collision radius (m)

# Per-agent parameters (0.0 / empty = use global default)
float64 vision_range         # 0.0 = use global default
float64 vision_fov           # 0.0 = use global default
float64 relaxation_time      # 0.0 = use global default
float64 repulsion_strength   # 0.0 = use global default
float64 repulsion_range      # 0.0 = use global default
string agent_type            # empty = use global default

arena_humansim_msgs/Waypoints waypoints
