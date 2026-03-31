# Shared interaction contract for the transitive interaction model
uint8 type
int32 min_participants
int32 max_participants   # -1 for unbounded
int32[] current_participants
int32 remaining_slots    # derived: max - len(current), -1 if unbounded
