import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _maybe_rviz(context, *args, **kwargs):
    rviz_val = LaunchConfiguration('rviz').perform(context)
    mode_val = LaunchConfiguration('mode').perform(context)

    if mode_val != 'master' or rviz_val.lower() == 'false':
        return []

    if rviz_val.lower() == 'true':
        pkg_share = get_package_share_directory('arena_humansim')
        config = os.path.join(pkg_share, 'config', 'arena_humansim.rviz')
    else:
        config = rviz_val

    return [ExecuteProcess(
        cmd=['rviz2', '-d', config],
        output='screen',
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='master', choices=['master', 'subsystem']),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('markers', default_value='0', description='0=off, 1=basic, 2=full'),
        DeclareLaunchArgument('rviz', default_value='true', description='true = default config, false = off, path = custom config'),
        Node(
            package='arena_humansim',
            executable='arena_humansim_node',
            name='arena_humansim',
            parameters=[
                {'mode': LaunchConfiguration('mode'),
                 'use_sim_time': LaunchConfiguration('use_sim_time'),
                 'publish_markers': LaunchConfiguration('markers')},
            ],
            output='screen',
        ),
        OpaqueFunction(function=_maybe_rviz),
    ])
