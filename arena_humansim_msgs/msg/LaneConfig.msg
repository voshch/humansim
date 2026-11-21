# Pedestrian lane of one source; empty segments = no lane
arena_humansim_msgs/LaneSegment[] segments
string side              # right, left or centre; required with segments
bool two_way             # shares its centreline with an opposing lane
bool has_offset          # true = use offset
float64 offset           # metres left of the centreline, if has_offset
float64 sigma_q          # 0 = default 0.5
float64 chord_tolerance  # 0 = default 0.05 m
float64 track_spacing    # 0 = derived
float64 arrival_radius   # 0 = derived
float64 merge_length     # 0 = derived
float64 spawn_length     # 0 = derived
float64 sweep_length     # 0 = derived
