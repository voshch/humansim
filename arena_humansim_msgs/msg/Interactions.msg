# Per-tick interaction state of the crowd engine: every live interaction plus
# the lifecycle edges fired this tick. Events are edge-triggered, record unthrottled.
std_msgs/Header header
arena_humansim_msgs/InteractionStatus[] interactions
arena_humansim_msgs/InteractionEvent[] events
