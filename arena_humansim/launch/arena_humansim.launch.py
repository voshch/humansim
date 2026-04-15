import os
import re
from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    SetLaunchConfiguration,
    Shutdown,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import ExecutableInPackage


def _compute_record_dir(context, *args, **kwargs):
    record = LaunchConfiguration("record").perform(context).lower() == "true"
    if not record:
        return [SetLaunchConfiguration("_record_dir", "")]
    rd = LaunchConfiguration("record_dir").perform(context)
    if not rd:
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        scenario = LaunchConfiguration("scenario").perform(context)
        mode = LaunchConfiguration("mode").perform(context)
        suffix = ""
        if scenario and mode == "master":
            stem = Path(scenario).stem
            slug = re.sub(r"[^\w.-]", "_", stem)
            if slug:
                suffix = f"_{slug}"
        rd = os.path.join(os.getcwd(), "recordings", f"{ts}{suffix}")
    return [SetLaunchConfiguration("_record_dir", rd)]


def _rviz_action(context, *args, **kwargs):
    rviz_val = LaunchConfiguration("rviz").perform(context)
    mode_val = LaunchConfiguration("mode").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)
    if mode_val != "master" or rviz_val.lower() == "false":
        return []
    if rviz_val.lower() == "true":
        config = os.path.join(get_package_share_directory("arena_humansim"), "config", "arena_humansim.rviz")
    else:
        config = rviz_val
    return [ExecuteProcess(
        cmd=["rviz2", "-d", config, "--ros-args", "-p", f"use_sim_time:={use_sim_time}"],
        output="screen",
    )]


def _renderer_action(context, *args, **kwargs):
    render = LaunchConfiguration("render").perform(context).lower() == "true"
    record = LaunchConfiguration("record").perform(context).lower() == "true"
    rd = LaunchConfiguration("_record_dir").perform(context)
    if not (render and record and rd):
        return [Shutdown(reason="node exited; no render requested")]
    fmt = LaunchConfiguration("render_format").perform(context)
    bag_dir = os.path.join(rd, "bag")
    output = os.path.join(rd, f"scenario.{fmt}")
    log_path = os.path.join(rd, "render.log")
    renderer_exe = ExecutableInPackage(package="arena_humansim", executable="arena_humansim_render").perform(context)
    return [ExecuteProcess(
        cmd=[renderer_exe, bag_dir, "--output", output, "--format", fmt, "--log-file", log_path],
        output="screen",
        on_exit=Shutdown(reason="renderer finished"),
    )]


def generate_launch_description():
    node = Node(
        package="arena_humansim",
        executable="arena_humansim_node",
        name="arena_humansim",
        parameters=[
            {"mode": LaunchConfiguration("mode"),
             "use_sim_time": LaunchConfiguration("use_sim_time"),
             "publish_markers": LaunchConfiguration("markers"),
             "record_bag": LaunchConfiguration("record"),
             "record_dir": LaunchConfiguration("_record_dir"),
             "scenario": LaunchConfiguration("scenario"),
             "ticks": LaunchConfiguration("ticks"),
             "time": ParameterValue(LaunchConfiguration("time"), value_type=float),
             "rtf": ParameterValue(LaunchConfiguration("rtf"), value_type=float)},
        ],
        output="screen",
    )

    kill_rviz = ExecuteProcess(cmd=["pkill", "-INT", "-f", "rviz2.*arena_humansim.rviz"], output="log")

    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="master", choices=["master", "subsystem"]),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("markers", default_value="0", description="0=off, 1=basic, 2=full"),
        DeclareLaunchArgument("rviz", default_value="true", description="true = default config, false = off, path = custom config"),
        DeclareLaunchArgument("record", default_value="false", description="record scenario to rosbag"),
        DeclareLaunchArgument("record_dir", default_value="", description="output dir (empty = ./recordings/<ts>[_<scenario>]/ relative to cwd)"),
        DeclareLaunchArgument("render", default_value="true", description="render video after node exits"),
        DeclareLaunchArgument("render_format", default_value="mp4", choices=["mp4", "gif"]),
        DeclareLaunchArgument("scenario", default_value="", description="scenario name or absolute YAML path (master mode only)"),
        DeclareLaunchArgument("ticks", default_value="0", description="stop after N ticks (0 = run until Ctrl-C)"),
        DeclareLaunchArgument("time", default_value="0.0", description="stop after N seconds of sim time (ignored if ticks is set)"),
        DeclareLaunchArgument("rtf", default_value="0.0", description="real-time factor (0 = unthrottled, 1.0 = real-time)"),
        OpaqueFunction(function=_compute_record_dir),
        node,
        OpaqueFunction(function=_rviz_action),
        RegisterEventHandler(OnProcessExit(
            target_action=node,
            on_exit=[kill_rviz, OpaqueFunction(function=_renderer_action)],
        )),
    ])
