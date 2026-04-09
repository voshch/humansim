from .block import BlockNodeSchema, BlockNode
from .follow import FollowNodeSchema, FollowNode
from .go_to import GoToNodeSchema, GoToNode
from .group_talk import GroupTalkNodeSchema, GroupTalkNode
from .idle import IdleNodeSchema, IdleNode
from .lie_on import LieOnNodeSchema, LieOnNode
from .look_at import LookAtNodeSchema, LookAtNode
from .queue import QueueNodeSchema, QueueNode
from .sit_on import SitOnNodeSchema, SitOnNode
from .talk_to import TalkToNodeSchema, TalkToNode
from .wave_at import WaveAtNodeSchema, WaveAtNode

__all__ = [
    "BlockNodeSchema",
    "BlockNode",
    "FollowNodeSchema",
    "FollowNode",
    "GoToNodeSchema",
    "GoToNode",
    "GroupTalkNodeSchema",
    "GroupTalkNode",
    "IdleNodeSchema",
    "IdleNode",
    "LieOnNodeSchema",
    "LieOnNode",
    "LookAtNodeSchema",
    "LookAtNode",
    "QueueNodeSchema",
    "QueueNode",
    "SitOnNodeSchema",
    "SitOnNode",
    "TalkToNodeSchema",
    "TalkToNode",
    "WaveAtNodeSchema",
    "WaveAtNode",
]

schemas = [
    BlockNodeSchema,
    FollowNodeSchema,
    GoToNodeSchema,
    GroupTalkNodeSchema,
    IdleNodeSchema,
    LieOnNodeSchema,
    LookAtNodeSchema,
    QueueNodeSchema,
    SitOnNodeSchema,
    TalkToNodeSchema,
    WaveAtNodeSchema,
]
