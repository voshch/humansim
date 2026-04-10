from glob import glob

from setuptools import find_packages, setup

package_name = "arena_humansim"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml") + glob("config/*.md") + glob("config/*.rviz")),
        ("share/" + package_name + "/config/agent_types", glob("config/agent_types/*.yaml")),
        ("share/" + package_name + "/config/benchmark", glob("config/benchmark/*.yaml")),
        ("share/" + package_name + "/config/scenarios", glob("config/scenarios/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=[
        "setuptools",
        "attrs",
        "cattrs",
        "pyyaml",
        "numpy",
        "scipy",
        "py_trees",
        "pyastar2d",
        "pydantic",
    ],
    zip_safe=True,
    maintainer="voshch",
    maintainer_email="dev@voshch.dev",
    description="Arena human simulator: modular, deterministic pedestrian simulation",
    license="MIT",
    entry_points={
        "console_scripts": [
            "arena_humansim_node = arena_humansim.manager.agent_manager:main",
            "benchmark = arena_humansim.benchmark:main",
        ],
    },
)
