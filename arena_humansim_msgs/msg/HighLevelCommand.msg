# High-level command emitted by BehaviorEngine per agent per tick
int32 agent_id
uint8 type
geometry_msgs/Pose2D target_pose
float64 desired_velocity
int32 interaction_target   # -1 for broadcast
int32 interaction_type
