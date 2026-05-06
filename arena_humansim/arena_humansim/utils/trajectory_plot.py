from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

_log = logging.getLogger("arena_humansim_render")


def _setup_logging(log_file: Path | None) -> None:
    fmt = logging.Formatter("[%(levelname)s] [%(name)s]: %(message)s")
    _log.setLevel(logging.INFO)
    _log.handlers.clear()
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    _log.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(fmt)
        _log.addHandler(fh)


@dataclass
class Frame:
    t_ns: int
    agents: list[tuple[int, float, float, float, float, float, str, int]]


@dataclass
class Geometry:
    walls: list[tuple[tuple[float, float], tuple[float, float]]]
    obstacles: list[tuple[tuple[float, float, float], tuple[float, float, float, float]]]
    world_objects: list[tuple[str, str, float, float]]


def _read_bag(bag_dir: Path) -> tuple[Geometry, list[Frame]]:
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    type_map = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}

    geometry = Geometry(walls=[], obstacles=[], world_objects=[])
    frames: list[Frame] = []

    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        msg_type = type_map.get(topic)
        if msg_type is None:
            continue
        msg = deserialize_message(raw, msg_type)

        if topic.endswith("world_geometry"):
            geometry.walls = [((s.x, s.y), (e.x, e.y)) for s, e in zip(msg.wall_starts, msg.wall_ends)]
            geometry.obstacles = [((o.pose.x, o.pose.y, o.pose.theta), (o.bb_x_min, o.bb_x_max, o.bb_y_min, o.bb_y_max)) for o in msg.obstacles]
            geometry.world_objects = [(o.object_id, o.type, o.pose.x, o.pose.y) for o in msg.world_objects]
        elif topic.endswith("agent_states"):
            agents = [(int(a.agent_id), a.pose.x, a.pose.y, a.pose.theta, a.velocity.x, a.velocity.y, a.policy or "", int(a.kind)) for a in msg.agents]
            frames.append(Frame(t_ns=t_ns, agents=agents))

    return geometry, frames


def _compute_bounds(geometry: Geometry, frames: list[Frame]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for (sx, sy), (ex, ey) in geometry.walls:
        xs.extend([sx, ex])
        ys.extend([sy, ey])
    for (px, py, _), (x_min, x_max, y_min, y_max) in geometry.obstacles:
        xs.extend([px + x_min, px + x_max])
        ys.extend([py + y_min, py + y_max])
    for _, _, x, y in geometry.world_objects:
        xs.append(x)
        ys.append(y)
    for frame in frames:
        for _, x, y, *_ in frame.agents:
            xs.append(x)
            ys.append(y)
    if not xs:
        return -10.0, 10.0, -10.0, 10.0
    pad = 1.0
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def render_trajectories(
    bag_dir: Path,
    output: Path,
    figure_title: str,
    dpi: int = 150,
    agent_radius: float = 0.35,
) -> None:
    """
    Render and export one scenario to a transparent background PNG file
    """
    import matplotlib.cm as cm

    _log.info(f"[trajectory] reading bag {bag_dir}")
    geometry, frames = _read_bag(bag_dir)
    frames.sort(key=lambda f: f.t_ns)

    if not frames:
        raise RuntimeError("no frames found")

    x_min, x_max, y_min, y_max = _compute_bounds(geometry, frames)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_facecolor("#fcfcfc")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.2)

    for (sx, sy), (ex, ey) in geometry.walls:
        ax.plot([sx, ex], [sy, ey], color="#333333", linewidth=2, zorder=1)

    # --- Build segments the same way render() processes frames ---
    # A new segment starts whenever an agent was absent in the previous frame.
    # This mirrors render()'s per-frame presence check exactly.

    # frame_index -> set of agent ids present
    presence: list[set[int]] = []
    for f in frames:
        presence.append({aid for aid, *_ in f.agents})

    # per-agent: list of segments, each segment is a list of (frame_idx, x, y)
    agent_segments: dict[int, list[list[tuple[int, float, float]]]] = {}
    agent_kinds: dict[int, int] = {}

    for fi, f in enumerate(frames):
        agents_in_frame = {aid: (x, y, kind) for aid, x, y, _, _, _, _, kind in f.agents}

        for aid, (x, y, kind) in agents_in_frame.items():
            agent_kinds[aid] = kind

            if aid not in agent_segments:
                agent_segments[aid] = [[(fi, x, y)]]
            else:
                # Was this agent present in the immediately preceding frame?
                was_present = aid in presence[fi - 1] if fi > 0 else True
                if not was_present:
                    # Absent for at least one frame → new segment, same as a
                    # render() frame where the agent simply wasn't drawn
                    agent_segments[aid].append([(fi, x, y)])
                else:
                    agent_segments[aid][-1].append((fi, x, y))

    # --- Plotting ---
    n_frames = len(frames)
    human_cmap = cm.get_cmap("Blues")

    for aid, segments in agent_segments.items():
        for segment in segments:
            if len(segment) < 2:
                continue

            frame_indices = [s[0] for s in segment]
            pts = np.array([(s[1], s[2]) for s in segment])

            if agent_kinds[aid] != 1:  # humans
                n_points = len(pts)
                for i in range(n_points - 1):
                    progress = frame_indices[i] / n_frames
                    seg_color = human_cmap(0.3 + progress * 0.7)
                    alpha_val = 0.1 + progress * 0.4
                    ax.plot(
                        pts[i : i + 2, 0],
                        pts[i : i + 2, 1],
                        color=seg_color,
                        linewidth=1.5,
                        alpha=alpha_val,
                        solid_capstyle="round",
                        zorder=3,
                    )
            else:  # robots
                ax.plot(pts[:, 0], pts[:, 1], color="white", linewidth=5, alpha=0.7, zorder=9)
                ax.plot(pts[:, 0], pts[:, 1], color="#d62728", linewidth=3.0, solid_capstyle="round", zorder=10)
                ax.scatter(pts[0, 0], pts[0, 1], color="#2ca02c", s=40, edgecolor="white", zorder=11)
                ax.scatter(pts[-1, 0], pts[-1, 1], color="black", marker="x", s=50, zorder=11)

    # --- Colorbar ---
    # sm = plt.cm.ScalarMappable(cmap=human_cmap, norm=plt.Normalize(vmin=0, vmax=n_frames))
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    # cbar.set_label("Simulation Step", rotation=270, labelpad=15, fontsize=10)
    # cbar.ax.tick_params(labelsize=8)

    human_ids = [aid for aid, k in agent_kinds.items() if k != 1]
    # legend_elements = [
    #     Line2D([0], [0], color="#d62728", lw=3, label="Robot"),
    #     Line2D([0], [0], color="gray", lw=1.5, alpha=0.5, label="Human"),
    #     Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", label="Start"),
    #     Line2D([0], [0], marker="x", color="black", label="End"),
    # ]
    # ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.15, 1.0), borderaxespad=0.0, fontsize="small", frameon=False)

    ax.set_title(f"{figure_title} (Humans: {len(human_ids)})", fontsize=20)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    output = output.with_suffix(".png")
    plt.savefig(output, dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    _log.info(f"[trajectory] wrote {output}")


def render_all_scenarios(
    scenario_dirs: list[tuple[str, Path]],
    output: Path,
    dpi: int = 150,
    agent_radius: float = 0.35,
) -> None:
    """
    NOTE: For now this is only used to render the simulation step color bar ONLY
    Render all scenarios in a grid layout with shared colorbar and legend.
    Grid is configured as 2 rows with dynamic columns to fit all scenarios.
    Publication-quality layout suitable for top-tier conference papers.

    Args:
        scenario_dirs: List of (scenario_name, bag_dir) tuples
        output: Output PNG file path
        dpi: Resolution in dots per inch (150 for print quality)
        agent_radius: Radius of agent circles (unused but kept for API compatibility)
    """
    import matplotlib.cm as cm
    from matplotlib.gridspec import GridSpec

    if not scenario_dirs:
        raise RuntimeError("no scenarios provided")

    n_scenarios = len(scenario_dirs)
    n_rows = 2
    n_cols = (n_scenarios + n_rows - 1) // n_rows  # Ceiling division

    _log.info(f"[all_scenarios] rendering {n_scenarios} scenarios in {n_rows}x{n_cols} grid")

    # Read all bag data upfront
    all_data: list[tuple[str, Geometry, list[Frame]]] = []
    global_x_min, global_x_max = float('inf'), float('-inf')
    global_y_min, global_y_max = float('inf'), float('-inf')

    for scenario_name, bag_dir in scenario_dirs:
        _log.info(f"[all_scenarios] reading {scenario_name}...")
        geometry, frames = _read_bag(bag_dir)
        frames.sort(key=lambda f: f.t_ns)

        if not frames:
            _log.warning(f"[all_scenarios] no frames found for {scenario_name}")
            continue

        all_data.append((scenario_name, geometry, frames))

        # Compute bounds for this scenario and update global bounds
        x_min, x_max, y_min, y_max = _compute_bounds(geometry, frames)
        global_x_min = min(global_x_min, x_min)
        global_x_max = max(global_x_max, x_max)
        global_y_min = min(global_y_min, y_min)
        global_y_max = max(global_y_max, y_max)

    if not all_data:
        raise RuntimeError("no valid scenarios found")

    # Use global bounds for all subplots (for fair comparison and visual consistency)
    x_min, x_max = global_x_min, global_x_max
    y_min, y_max = global_y_min, global_y_max

    # Calculate figure size for publication quality
    # Each subplot sized for clarity while keeping overall figure manageable
    subplot_width = 3.5
    subplot_height = 3.0
    fig_width = n_cols * subplot_width + 1.2  # Extra space for colorbar and legend
    fig_height = n_rows * subplot_height + 0.5  # Extra space for title padding

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

    # Create grid with space for colorbar and legend on the right
    gs = GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        left=0.08,
        right=0.80,
        top=0.96,
        bottom=0.08,
        wspace=0.25,  # Horizontal spacing between subplots
        hspace=0.30,  # Vertical spacing between subplots
    )

    # Initialize all axes
    # axes = []
    # for i in range(n_rows):
    #     row_axes = []
    #     for j in range(n_cols):
    #         ax = fig.add_subplot(gs[i, j])
    #         ax.set_xlim(x_min, x_max)
    #         ax.set_ylim(y_min, y_max)
    #         ax.set_aspect("equal")
    #         ax.set_facecolor("#fcfcfc")  # Light background for print
    #         ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.2)
    #         # Clean up axis appearance
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['right'].set_visible(False)
    #         ax.spines['left'].set_linewidth(0.5)
    #         ax.spines['bottom'].set_linewidth(0.5)
    #         row_axes.append(ax)
    #     axes.append(row_axes)

    # Plotting
    human_cmap = cm.get_cmap("Blues")
    max_frames_overall = max(len(frames) for _, _, frames in all_data)

    # scenario_idx = 0
    # for i in range(n_rows):
    #     for j in range(n_cols):
    #         if scenario_idx >= len(all_data):
    #             # Hide unused subplots
    #             axes[i][j].set_visible(False)
    #             scenario_idx += 1
    #             continue

    #         scenario_name, geometry, frames = all_data[scenario_idx]
    #         ax = axes[i][j]

    #         # Draw walls
    #         for (sx, sy), (ex, ey) in geometry.walls:
    #             ax.plot([sx, ex], [sy, ey], color="#333333", linewidth=1.5, zorder=1)

    #         # Build segments (same logic as render_trajectories)
    #         presence: list[set[int]] = []
    #         for f in frames:
    #             presence.append({aid for aid, *_ in f.agents})

    #         agent_segments: dict[int, list[list[tuple[int, float, float]]]] = {}
    #         agent_kinds: dict[int, int] = {}
    #         human_count = 0

    #         for fi, f in enumerate(frames):
    #             agents_in_frame = {aid: (x, y, kind) for aid, x, y, _, _, _, _, kind in f.agents}

    #             for aid, (x, y, kind) in agents_in_frame.items():
    #                 agent_kinds[aid] = kind
    #                 if kind != 1:
    #                     human_count = len([k for k in agent_kinds.values() if k != 1])

    #                 if aid not in agent_segments:
    #                     agent_segments[aid] = [[(fi, x, y)]]
    #                 else:
    #                     was_present = aid in presence[fi - 1] if fi > 0 else True
    #                     if not was_present:
    #                         agent_segments[aid].append([(fi, x, y)])
    #                     else:
    #                         agent_segments[aid][-1].append((fi, x, y))

    #         # Plot trajectories using consistent coloring across all subplots
    #         n_frames = len(frames)
    #         for aid, segments in agent_segments.items():
    #             for segment in segments:
    #                 if len(segment) < 2:
    #                     continue

    #                 frame_indices = [s[0] for s in segment]
    #                 pts = np.array([(s[1], s[2]) for s in segment])

    #                 if agent_kinds[aid] != 1:  # humans
    #                     n_points = len(pts)
    #                     for idx in range(n_points - 1):
    #                         # Use global frame count for consistent coloring across subplots
    #                         progress = frame_indices[idx] / max_frames_overall
    #                         seg_color = human_cmap(0.3 + progress * 0.7)
    #                         alpha_val = 0.1 + progress * 0.4
    #                         ax.plot(
    #                             pts[idx : idx + 2, 0],
    #                             pts[idx : idx + 2, 1],
    #                             color=seg_color,
    #                             linewidth=1.2,
    #                             alpha=alpha_val,
    #                             solid_capstyle="round",
    #                             zorder=3,
    #                         )
    #                 else:  # robots
    #                     ax.plot(pts[:, 0], pts[:, 1], color="white", linewidth=4, alpha=0.7, zorder=9)
    #                     ax.plot(pts[:, 0], pts[:, 1], color="#d62728", linewidth=2.5, solid_capstyle="round", zorder=10)
    #                     ax.scatter(pts[0, 0], pts[0, 1], color="#2ca02c", s=35, edgecolor="white", zorder=11)
    #                     ax.scatter(pts[-1, 0], pts[-1, 1], color="black", marker="x", s=40, zorder=11)

    #         # Subplot title with agent count
    #         ax.set_title(
    #             f"{scenario_name}\n({human_count} humans)",
    #             fontsize=11,
    #             fontweight="normal",
    #             pad=8,
    #         )

    #         # Clean tick appearance for publication
    #         ax.tick_params(labelsize=8, length=3, width=0.5)

    #         scenario_idx += 1

    # Shared colorbar (positioned on the right)
    cbar_ax = fig.add_axes([0.15, 0.82, 0.75, 0.02])

    sm = plt.cm.ScalarMappable(cmap=human_cmap, norm=plt.Normalize(vmin=0, vmax=max_frames_overall))
    sm.set_array([])

    # # Ensure orientation is set to 'horizontal'
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')

    # # Adjust labelpad as '18' might be too far for a horizontal bar
    cbar.set_label("Simulation Step", labelpad=10, fontsize=16, fontweight="normal")
    cbar.ax.tick_params(labelsize=8)

    # Save with transparency as requested
    fig.savefig(output.with_name(f"{output.stem}_colorbar.png"), dpi=dpi, transparent=True, bbox_inches='tight')
    return

    # Shared legend (positioned on the right below colorbar)
    # legend_elements = [
    #     Line2D([0], [0], color="#d62728", lw=2.5, label="Robot"),
    #     Line2D([0], [0], color="gray", lw=1.2, alpha=0.5, label="Human"),
    #     Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=6, label="Start"),
    #     Line2D([0], [0], marker="x", color="black", markersize=7, label="End"),
    # ]
    # legend = fig.legend(
    #     handles=legend_elements,
    #     loc="lower center",  # Changed from lower right to center
    #     bbox_to_anchor=(0.5, 0.03),  # Centered horizontally (0.5)
    #     ncols=4,  # Forces all 4 items into one row
    #     columnspacing=1.5,  # Adds breathing room between items
    #     fontsize=9,
    #     frameon=True,
    #     fancybox=False,
    #     edgecolor="#cccccc",
    #     framealpha=0.95,
    # )
    # Save with tight bounding box to ensure the PNG only contains the horizontal strip
    fig.savefig(output.with_name(f"{output.stem}_legend.png"), dpi=dpi, transparent=True, bbox_inches='tight')

    fig.suptitle("Agent Trajectories - All Scenarios", fontsize=14, fontweight="bold", y=0.98)

    output = output.with_suffix(".png")
    plt.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _log.info(f"[all_scenarios] wrote {output}")


def main():
    """
    Find all the scenarios run in `/opt/arena_ws/src/recordings`, export each run into a PNG file
    """
    import os

    base_path = Path("/opt/arena_ws/src/recordings/")
    recorded_scenarios = os.listdir(base_path)

    scenario_dirs = []

    for scenario in recorded_scenarios:
        bag_dir = base_path / scenario / "bag"
        if not bag_dir.exists() or not bag_dir.is_dir():
            continue

        scenario_name = scenario.split("_")[1:]
        scenario_name = ' '.join(part.title() for part in scenario_name)
        print(scenario_name)
        scenario_dirs.append((scenario_name, bag_dir))

    # Combined figure
    output = base_path / "all_scenarios.png"
    render_all_scenarios(scenario_dirs, output)

    # Map `scenario_name` to figure title
    figure_title_mapping = {"Robot Corridor": "Robot Corridor", "Queue": "Queue", "Robot Simple Crossing": "Robot Simple Crossing", "Robot Test": "Multi Robot", "Bottleneck": "Bottleneck", "Robot Bottleneck": "Robot Bottleneck"}
    for name, bag_dir in scenario_dirs:
        figure_title = figure_title_mapping[name]

        print("Mapped", figure_title)
        render_trajectories(bag_dir, bag_dir.parent.parent / f"{name}.png", figure_title=figure_title)


if __name__ == "__main__":
    main()
