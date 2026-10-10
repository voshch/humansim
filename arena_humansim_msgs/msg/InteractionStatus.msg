# Snapshot of one live interaction (InteractionManager), for recording and analysis
uint8 FORMING=0
uint8 ACTIVE=1

int32 interaction_id
uint8 interaction_type       # arena_humansim InteractionType index
uint8 outcome                # FORMING | ACTIVE
int32[] participants
int32[] queue
bool arrived                 # every participant is on its formation slot (true without a formation)
bool holding                 # HOLD_ONSET has fired and the interaction has not ended
float64 hold_elapsed         # seconds counted against duration since the hold started
float64 duration             # hold duration in seconds, -1 = unbounded
