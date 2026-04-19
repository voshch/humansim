from arena_humansim.core.agents import ActionDef, NeedCondition
from arena_humansim.core.world_knowledge import WorldKnowledge
from arena_humansim.utils.types import NeedState


def check_condition(value: float, condition: NeedCondition) -> bool:
    if condition.below is not None and value >= condition.below:
        return False
    if condition.above is not None and value <= condition.above:
        return False
    return True


def preconditions_met(
    needs: dict[str, NeedState],
    when: dict[str, NeedCondition],
) -> bool:
    for need_name, condition in when.items():
        need = needs.get(need_name)
        if need is None:
            return False
        if not check_condition(need.value, condition):
            return False
    return True


def score_actions(
    needs: dict[str, NeedState],
    actions: dict[str, ActionDef],
    utility_weights: dict[str, float],
    world: WorldKnowledge,
) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []

    for name, action in actions.items():
        if not preconditions_met(needs, action.when):
            continue

        utility = 0.0
        for need_name, delta in action.satisfies.items():
            need = needs.get(need_name)
            if need is None:
                continue
            urgency = (100.0 - need.value) / 100.0
            weight = utility_weights.get(need_name, 1.0)
            utility += urgency * weight * (delta / 100.0)

        if action.target_object_id:
            q_len = world.queue_length_for_object(action.target_object_id)
            penalty = q_len * 0.05
            utility *= max(0.2, 1.0 - penalty)
        elif action.target_object_type:
            q_len = world.queue_length(action.target_object_type)
            penalty = q_len * 0.05
            utility *= max(0.2, 1.0 - penalty)

        if utility > 0.0:
            scored.append((name, utility))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
