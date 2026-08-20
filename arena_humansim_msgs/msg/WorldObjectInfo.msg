string object_id
string type
geometry_msgs/Pose2D pose
uint32 capacity
string[] satisfies_keys
float64[] satisfies_values
float64 interaction_radius     # 0.0 = use interaction-type default

# Optional formation (line | cluster | f_formation | dyad). Empty type = no formation.
string formation_type
string[] formation_param_keys
float64[] formation_param_values

# Seat poses (world frame): cluster formations on this object place participants exactly here. Empty = ring around the pose.
geometry_msgs/Pose2D[] seats
