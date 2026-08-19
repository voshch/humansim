# One active attention channel of an agent
string slot                  # head | arm | arm_l | arm_r | body
geometry_msgs/Point at       # target, world frame (aimed slots)
string clip                  # body: animation clip name
string hand                  # arm: dominant hand l | r, empty = default
