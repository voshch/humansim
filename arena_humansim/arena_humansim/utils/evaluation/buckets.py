DRIVER_CLASS = {
    "sfm": "classical",
    "hsfm": "classical",
    "orca": "classical",
    "straight": "classical",
    "nsp": "learned",
    "socialgail": "learned",
}

# Fine-grained taxonomy used by the within-class redundancy table (six drivers,
# four classes per the abstract). Distinct from DRIVER_CLASS, which the
# §4.4 K_nav/K_bt headline binarizes as classical-vs-learned.
DRIVER_CLASS_FINE = {
    "sfm": "force",
    "hsfm": "force",
    "orca": "geometric",
    "straight": "no_avoidance",
    "nsp": "learned",
    "socialgail": "learned",
}

SCENARIO_BUCKET = {
    "simple_crossing": "nav",
    "corridor": "nav",
    "flow_coverage": "nav",
    "t_junction": "nav",
    "bottleneck": "nav",
    "l_corridor": "nav",
    "escort": "bt",
    "queue": "bt",
    "bt_coverage": "bt",
    "robot_test": "het",
    "robot_simple_crossing": "nav",
    "robot_corridor": "nav",
    "robot_flow_coverage": "nav",
    "robot_t_junction": "nav",
    "robot_bottleneck": "nav",
    "robot_l_corridor": "nav",
    "robot_escort": "bt",
    "robot_queue": "bt",
    "robot_bt_coverage": "bt",
}

KNOWN_BUCKETS = ("nav", "bt", "het")
