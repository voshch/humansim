DRIVER_CLASS = {
    "sfm": "classical",
    "hsfm": "classical",
    "orca": "classical",
    "straight": "classical",
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
}

KNOWN_BUCKETS = ("nav", "bt", "het")
