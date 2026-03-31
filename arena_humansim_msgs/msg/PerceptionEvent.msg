# A single perception delta event for one agent
uint8 ENTERED=0
uint8 LEFT=1
uint8 UPDATED=2

int32 agent_id
uint8 event_type
geometry_msgs/Pose2D pose
float64 velocity_x
float64 velocity_y
