FROM ros:jazzy

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip \
      python3-colcon-common-extensions \
      ros-jazzy-rosbag2-storage-mcap \
      ros-jazzy-rviz2 \
      ffmpeg \
      git \
    && rm -rf /var/lib/apt/lists/*

ENV WS=/opt/arena_ws
WORKDIR $WS/src
COPY arena_humansim arena_humansim
COPY arena_humansim_msgs arena_humansim_msgs

# All Python deps from setup.py. The `test` extra transitively pulls
# socialgail, nsp, and robot (sarl, dsrnn, drlvo, cadrl) — i.e. everything.
RUN pip3 install --break-system-packages --ignore-installed numpy && \
    pip3 install --break-system-packages -e "$WS/src/arena_humansim[test]"

WORKDIR $WS
RUN source /opt/ros/jazzy/setup.bash && colcon build --symlink-install

ENV ARENA_DATA_DIR=/data
RUN mkdir -p /data && \
    echo 'source /opt/ros/jazzy/setup.bash' >> /etc/bash.bashrc && \
    echo 'source /opt/arena_ws/install/setup.bash' >> /etc/bash.bashrc

CMD ["bash"]
