This repo is for the simulation and verification of *Task and Motion Planning for Dual-Arm Robotic Bin Packing*.

This repo is derived from [Fukuda's Isaac Sim Env](https://github.com/yamakita-nec/isaac_sim_environment.git). The original repo is based on Isaac Sim 4.0 and Omniverse Launcher (deprecated sicne Oct. 1, 2025), which is a bit outdated and not well supported anymore. Thus, this repo moves to Isaac Sim 4.5, which is Python 3.10-based and well suited for Ubuntu22.04 + ROS2 Humble.

# simulator

**Notice**: A lot of scripts were directly copied from the original repo. Only binpacking_environment.py is modified for migrating to Isaac Sim 4.5.

## Set up Procedure
1. Use poetry install
```bash
poetry install
```
This will automatically install all necessary libraries including Isaac Sim 4.5.

About installation of Isaac Sim 4.5: Isaac Sim 4.5 supports both binary installation and pip installation. For Python development, pip installation will make your life easier, saving efforts of setting up environment. For further details, please refer to [official doc](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_python.html).

2. Activate virtual environment using poetry
```bash
eval $(poetry env activate)
```

3. Configure ROS2 environment
```bash
source /opt/ros/humble/setup.bash
source ~/IsaacSim-ros_workspaces/humble_ws/install/local_setup.sh
```
The second line is for using ROS2 in Isaac Sim, please refer to [official doc](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_ros.html), chapter Setting Up Workspaces.

## Usage
In src/simulator, the python file bpp_ros2_sim_node.py provides an online simulation for verification of 3D BPP Algorithm + Sequential Dependency Extraction, acting as a server (the client is the planner), using ROS2 as communication interface. To run this file, simply use:
```bash
python simulator/src/simulator/bpp_ros2_sim_node.py 
```

Note:
1. Boexes in the simulation are purely visual objects since dynamical behaviours (dual-arm) haven't been introduced.
2. There are bugs in the task scheduler (dead lock)

