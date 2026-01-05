import os
import sys

# 仮想環境が有効化されているかチェック
# if 'VIRTUAL_ENV' not in os.environ:
#     print(f"Error: Virtual environment is not activated.", file=sys.stderr)
#     sys.exit(1)

import re
import time
from typing import Union
import numpy as np
from pathlib import Path

from isaacsim import SimulationApp

FRANKA_STAGE_PATH = "/Franka"
FRANKA_USD_PATH = "/Isaac/Robots/Franka/franka_alt_fingers.usd"
CAMERA_PRIM_PATH = f"{FRANKA_STAGE_PATH}/panda_hand/geometry/realsense/realsense_camera"
BACKGROUND_STAGE_PATH = "/background"
# BACKGROUND_USD_PATH = "/Isaac/Environments/Simple_Room/simple_room.usd"
# BACKGROUND_USD_PATH = "/Isaac/Environments/Grid/default_environment.usd"
BACKGROUND_USD_PATH = "/Isaac/Environments/Grid/gridroom_curved.usd"
GRAPH_PATH = "/ActionGraph"
REALSENSE_VIEWPORT_NAME = "RealSense"

CONFIG = {"renderer": "RayTracedLighting", "headless": False}

# Example ROS2 bridge sample demonstrating the manual loading of stages
# and creation of ROS components

simulation_app = SimulationApp(CONFIG)

# More imports that need to compare after we create the app
import carb
from omni.isaac.core import SimulationContext  # noqa E402
from omni.isaac.core.utils.prims import set_targets
from omni.isaac.core.utils import (  # noqa E402
    extensions,
    nucleus,
    prims,
    rotations,
    stage,
    viewports,
)
from omni.isaac.core_nodes.scripts.utils import set_target_prims  # noqa E402
from pxr import Gf, UsdGeom, Sdf  # noqa E402
import omni.graph.core as og  # noqa E402
import omni
import omni.ui
import omni.usd


# enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
extensions.enable_extension("omni.isaac.repl")

simulation_context = SimulationContext(stage_units_in_meters=1.0)

# Locate Isaac Sim assets folder to load environment and robot stages
assets_root_path = nucleus.get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

# Preparing stage
viewports.set_camera_view(eye=np.array([1.2, 1.2, 0.8]), target=np.array([0, 0, 0.5]))

# Loading the simple_room environment
stage.add_reference_to_stage(
    assets_root_path + BACKGROUND_USD_PATH, BACKGROUND_STAGE_PATH
)

# Add a dome light to the world
current_stage = stage.get_current_stage()
dome_light = current_stage.DefinePrim("/World/DomeLight", "DomeLight")
dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(900.0)

# Loading the franka robot USD
prims.create_prim(
    FRANKA_STAGE_PATH,
    "Xform",
    position=np.array([0, -0.64, 0]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)),
    usd_path=assets_root_path + FRANKA_USD_PATH,
)

# add some objects, spread evenly along the X axis
# with a fixed offset from the robot in the Y and Z
prims.create_prim(
    "/cracker_box",
    "Xform",
    position=np.array([-0.2, -0.25, 0.15]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
)
prims.create_prim(
    "/sugar_box",
    "Xform",
    position=np.array([-0.07, -0.25, 0.1]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 1, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
)
prims.create_prim(
    "/soup_can",
    "Xform",
    position=np.array([0.1, -0.25, 0.10]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
)
prims.create_prim(
    "/mustard_bottle",
    "Xform",
    position=np.array([0.0, 0.15, 0.12]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
    usd_path=assets_root_path
    + "/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
)



from components.widget import StagePreviewWidget
rs_viewport = omni.ui.Window(REALSENSE_VIEWPORT_NAME, width=1280, height=720+20)
with rs_viewport.frame:
    rs_widget = StagePreviewWidget(camera_path = CAMERA_PRIM_PATH, resolution = (1280, 720))
rs_api = rs_widget.viewport_api


simulation_app.update()

try:
    ros_domain_id = int(os.environ["ROS_DOMAIN_ID"])
    print("Using ROS_DOMAIN_ID: ", ros_domain_id)
except ValueError:
    print("Invalid ROS_DOMAIN_ID integer value. Setting value to 0")
    ros_domain_id = 0
except KeyError:
    print("ROS_DOMAIN_ID environment variable is not set. Setting value to 0")
    ros_domain_id = 0

# Enable Eco Mode
import carb.settings
# carb.settings.get_settings().get_settings_dictionary("/rtx/ecoMode")
carb.settings.get_settings().set_bool("/rtx/ecoMode/enabled", True)


# Creating a action graph with ROS component nodes
try:
    og.Controller.edit(
        {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                ("ReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                ("Context", "omni.isaac.ros2_bridge.ROS2Context"),
                ("PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                (
                    "SubscribeJointState",
                    "omni.isaac.ros2_bridge.ROS2SubscribeJointState",
                ),
                (
                    "ArticulationController",
                    "omni.isaac.core_nodes.IsaacArticulationController",
                ),
                ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("RunSimulationFrame", "omni.isaac.core_nodes.OgnIsaacRunOneSimulationFrame"),
                # ("renderProduct", "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
                ("cameraHelperRgb", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("cameraHelperInfo", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("cameraHelperDepth", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
                (
                    "OnImpulseEvent.outputs:execOut",
                    "ArticulationController.inputs:execIn",
                ),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                (
                    "ReadSimTime.outputs:simulationTime",
                    "PublishJointState.inputs:timeStamp",
                ),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                (
                    "SubscribeJointState.outputs:jointNames",
                    "ArticulationController.inputs:jointNames",
                ),
                (
                    "SubscribeJointState.outputs:positionCommand",
                    "ArticulationController.inputs:positionCommand",
                ),
                (
                    "SubscribeJointState.outputs:velocityCommand",
                    "ArticulationController.inputs:velocityCommand",
                ),
                (
                    "SubscribeJointState.outputs:effortCommand",
                    "ArticulationController.inputs:effortCommand",
                ),
                ("OnTick.outputs:tick", "RunSimulationFrame.inputs:execIn"),
                # ("RunSimulationFrame.outputs:step", "renderProduct.inputs:execIn"),
                ("RunSimulationFrame.outputs:step", "cameraHelperRgb.inputs:execIn"),
                ("RunSimulationFrame.outputs:step", "cameraHelperInfo.inputs:execIn"),
                ("RunSimulationFrame.outputs:step", "cameraHelperDepth.inputs:execIn"),
                # ("renderProduct.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                # ("renderProduct.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
                # ("renderProduct.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                ("Context.outputs:context", "cameraHelperRgb.inputs:context"),
                ("Context.outputs:context", "cameraHelperInfo.inputs:context"),
                ("Context.outputs:context", "cameraHelperDepth.inputs:context"),
                # (
                #     "renderProduct.outputs:renderProductPath",
                #     "cameraHelperRgb.inputs:renderProductPath",
                # ),
                # (
                #     "renderProduct.outputs:renderProductPath",
                #     "cameraHelperInfo.inputs:renderProductPath",
                # ),
                # (
                #     "renderProduct.outputs:renderProductPath",
                #     "cameraHelperDepth.inputs:renderProductPath",
                # ),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("Context.inputs:domain_id", ros_domain_id),
                # Setting the /Franka target prim to Articulation Controller node
                ("ArticulationController.inputs:robotPath", FRANKA_STAGE_PATH),
                ("PublishJointState.inputs:topicName", "isaac_joint_states"),
                ("SubscribeJointState.inputs:topicName", "isaac_joint_commands"),
                # ("renderProduct.inputs:cameraPrim", CAMERA_PRIM_PATH),
                ("cameraHelperRgb.inputs:frameId", "sim_camera"),
                ("cameraHelperRgb.inputs:topicName", "rgb"),
                ("cameraHelperRgb.inputs:type", "rgb"),
                ("cameraHelperRgb.inputs:renderProductPath", rs_api.render_product_path),
                ("cameraHelperInfo.inputs:frameId", "sim_camera"),
                ("cameraHelperInfo.inputs:topicName", "camera_info"),
                ("cameraHelperInfo.inputs:type", "camera_info"),
                ("cameraHelperInfo.inputs:renderProductPath", rs_api.render_product_path),
                ("cameraHelperDepth.inputs:frameId", "sim_camera"),
                ("cameraHelperDepth.inputs:topicName", "depth"),
                ("cameraHelperDepth.inputs:type", "depth"),
                ("cameraHelperDepth.inputs:renderProductPath", rs_api.render_product_path),
            ],
        },
    )
except Exception as e:
    print(e)


# # Setting the /Franka target prim to Publish JointState node
set_target_prims(
    primPath="/ActionGraph/PublishJointState", targetPrimPaths=[FRANKA_STAGE_PATH]
)

# # Fix camera settings since the defaults in the realsense model are inaccurate
realsense_prim = camera_prim = UsdGeom.Camera(
    stage.get_current_stage().GetPrimAtPath(CAMERA_PRIM_PATH)
)
realsense_prim.GetHorizontalApertureAttr().Set(20.955)
realsense_prim.GetVerticalApertureAttr().Set(15.7)
realsense_prim.GetFocalLengthAttr().Set(18.8)
realsense_prim.GetFocusDistanceAttr().Set(400)


# set_targets(
#     prim=stage.get_current_stage().GetPrimAtPath(GRAPH_PATH + "/setCamera"),
#     attribute="inputs:cameraPrim",
#     target_prim_paths=[CAMERA_PRIM_PATH],
# )

# from omni.kit.widget.viewport import ViewportWidget
# viewport_window = omni.ui.Window('SimpleViewport', width=1280, height=720+20)
# with viewport_window.frame:
#     viewport_widget = ViewportWidget(resolution = (1280, 720))
# viewport_api = viewport_widget.viewport_api
# viewport_api.resolution = (640, 480)
# viewport_api.camera_path = CAMERA_PRIM_PATH
# from components.widget import StagePreviewWidget
# rs_viewport = omni.ui.Window(REALSENSE_VIEWPORT_NAME, width=1280, height=720+20)
# with rs_viewport.frame:
#     rs_widget = StagePreviewWidget(camera_path = CAMERA_PRIM_PATH, resolution = (1280, 720))
# rs_api = rs_widget.viewport_api
# for helper in ["cameraHelperRgb", "cameraHelperInfo", "cameraHelperDepth"]:
#     set_targets(
#         prim=stage.get_current_stage().GetPrimAtPath(GRAPH_PATH + "/" + helper),
#         attribute="inputs:renderProductPath",
#         target_prim_paths=[rs_api.render_product_path],
#     )

# Dock the second camera window
viewport = omni.ui.Workspace.get_window("Viewport")
# rs_viewport = ViewportWindow(REALSENSE_VIEWPORT_NAME, width=1280, height=720+20)
# rs_viewport.viewport_api.camera_path = CAMERA_PRIM_PATH
simulation_app.update()
# time.sleep(2)
rs_viewport.dock_in(viewport, omni.ui.DockPosition.RIGHT)




# need to initialize physics getting any articulation..etc
simulation_context.initialize_physics()
simulation_context.get_physics_context().enable_gpu_dynamics(True)

simulation_context.play()


while simulation_app.is_running():

    # Run with a fixed step size
    simulation_context.step(render=True)

    # Tick the Publish/Subscribe JointState, Publish TF and Publish Clock nodes each frame
    og.Controller.set(
        og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"), True
    )

simulation_context.stop()
simulation_app.close()
