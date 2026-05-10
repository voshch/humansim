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
# Sec.4.4 K_nav/K_bt headline binarizes as classical-vs-learned.
DRIVER_CLASS_FINE = {
    "sfm": "force",
    "hsfm": "force",
    "orca": "geometric",
    "straight": "no_avoidance",
    "nsp": "learned",
    "socialgail": "learned",
}

SCENARIO_BUCKET = {
    # nav/sparse
    "nav_sparse_corridor":  "nav",   "robot_nav_sparse_corridor":  "nav",
    "nav_sparse_crossing":  "nav",   "robot_nav_sparse_crossing":  "nav",
    "nav_sparse_merge":     "nav",   "robot_nav_sparse_merge":     "nav",
    "nav_sparse_bend":      "nav",   "robot_nav_sparse_bend":      "nav",
    "nav_sparse_flow":      "nav",   "robot_nav_sparse_flow":      "nav",
    # nav/dense
    "nav_dense_corridor":   "nav",   "robot_nav_dense_corridor":   "nav",
    "nav_dense_crossing":   "nav",   "robot_nav_dense_crossing":   "nav",
    "nav_dense_bottleneck": "nav",   "robot_nav_dense_bottleneck": "nav",
    "nav_dense_flow":       "nav",   "robot_nav_dense_flow":       "nav",
    # bt/sparse
    "bt_sparse_group_conversation": "bt", "robot_bt_sparse_group_conversation": "bt",
    "bt_sparse_queue_use":          "bt", "robot_bt_sparse_queue_use":          "bt",
    "bt_sparse_service_static":     "bt", "robot_bt_sparse_service_static":     "bt",
    "bt_sparse_service_mobile":     "bt", "robot_bt_sparse_service_mobile":     "bt",
    "bt_sparse_pair":               "bt", "robot_bt_sparse_pair":               "bt",
    "bt_sparse_sit":                "bt", "robot_bt_sparse_sit":                "bt",
    "bt_sparse_needs":              "bt", "robot_bt_sparse_needs":              "bt",
    "bt_sparse_compound":           "bt", "robot_bt_sparse_compound":           "bt",
    # bt/dense
    "bt_dense_group_conversation":  "bt", "robot_bt_dense_group_conversation":  "bt",
    "bt_dense_queue_use":           "bt", "robot_bt_dense_queue_use":           "bt",
    "bt_dense_service_mobile":      "bt", "robot_bt_dense_service_mobile":      "bt",
    # het/dense (robot constitutive)
    "het_dense_mixed_speeds":          "het",
    "het_dense_wheelchair_bottleneck": "het",
    "het_dense_robot_in_group":        "het",
}

KNOWN_BUCKETS = ("nav", "bt", "het")
