#! /usr/bin/python3

from typing import Dict, List, Tuple
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.utils import extensions
# Extensions settings
extensions.enable_extension("omni.isaac.repl")

import os
import numpy as np
import numpy.typing as npt
import glob
import random
from dataclasses import dataclass
import trimesh

import carb
from omni.isaac.core import World
from omni.isaac.core.utils import (
  prims,
  stage,
)
from omni.physx.scripts import utils

import omni.kit.commands as commands
import omni.timeline
from pxr import Usd, UsdGeom, Sdf, Gf

###################### Constants ######################

PACKAGE_PATH = os.path.dirname(os.path.dirname(__file__)) # path to isaac_env
OBJECTS_PATH = os.path.join(PACKAGE_PATH, "objects", "ycb") 
BOX_SIZE_INNER_ORIGINAL = np.array([0.5, 0.5, 0.3])
BOX_SIZE_OUTER_ORIGINAL = np.array([0.52, 0.52, 0.32])
BOX_SCALE = np.array([ 0.4/0.5, 0.4/0.5, 0.2/0.3 ])
BOX_SIZE_INNER = BOX_SCALE * BOX_SIZE_INNER_ORIGINAL
BOX_SIZE_OUTER = BOX_SCALE * BOX_SIZE_OUTER_ORIGINAL


###################### Custom Classes ######################
@dataclass
class ItemInfo:
    prim: Usd.Prim # prim object
    usd_path: str  # (str) path to usd file
    obj_path: str  # (str) path to obj file
    size: np.ndarray  # (np.ndarray) size of the object
    bbox_vertices: np.ndarray # (np.ndarray) bounding box range of the object

    def move_to(self, position: Gf.Vec3d):
        assert self.prim is not None and type(self.prim) == Usd.Prim
        self.prim.GetAttribute("xformOp:translate").Set(position)
    
    def get_bbox_range(self) -> Tuple[List[float], List[float], List[float]]:
        # 1行目 -> x座標, 2行目 -> y座標, 3行目 -> z座標
        x_range = [np.min(self.bbox_vertices[:, 0]), np.max(self.bbox_vertices[:, 0])]
        y_range = [np.min(self.bbox_vertices[:, 1]), np.max(self.bbox_vertices[:, 1])]
        z_range = [np.min(self.bbox_vertices[:, 2]), np.max(self.bbox_vertices[:, 2])]
        return (x_range, y_range, z_range)
    

###################### Helper Functions ######################

def object_name_to_usd_path(object_name: str) -> str:
    return os.path.join(OBJECTS_PATH, object_name, "google_16k", "_converted", "textured_transformed_obj.usd")

def object_name_to_obj_path(object_name: str) -> str:
    return os.path.join(OBJECTS_PATH, object_name, "google_16k", "textured_transformed.obj")

def add_ycb_object(object_name: str, prim_name: str) -> ItemInfo:
    usd_path = object_name_to_usd_path(object_name)
    obj_path = object_name_to_obj_path(object_name)

    # Add object to stage
    prim_path = f"/World/Item/{prim_name}"
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"USD file not found: {usd_path}")
    prim = stage.add_reference_to_stage(usd_path, prim_path)
    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(4,4,4))
    utils.setRigidBody(prim, "sdf", False)

    # Analyze object data
    mesh: trimesh.base.Trimesh = trimesh.load(obj_path)
    center = mesh.bounding_box.centroid
    size = mesh.bounding_box.extents
    print(f"Object: {object_name}, Center: {center}, Size: {size}")

    # Store object data
    added_objects[prim_path] = ItemInfo(prim, usd_path, obj_path, size, mesh.bounding_box.vertices)
    return added_objects[prim_path]

def add_storage(position: Gf.Vec3d) -> Usd.Prim:
    file_name = "box_500_500_300.usd"
    usd_path = os.path.join(PACKAGE_PATH, "descriptions", "isaac", file_name)
    prim_path = f"/World/Storage"
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"USD file not found: {usd_path}")
    prim = stage.add_reference_to_stage(usd_path, prim_path)
    prim.GetAttribute("xformOp:translate").Set(position)
    prim.GetAttribute("xformOp:scale").Set(Gf.Vec3d(*BOX_SCALE))
    box_raw_prim = stage.get_current_stage().GetPrimAtPath("/World/Storage/box_500_500_300")
    box_raw_prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(0.70710677, 0.70710677, 0, 0))
    # prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(0.70710677, 0.70710677, 0, 0))
    utils.setCollider(prim, "sdf")
    return prim

def remove_object(prim_path: str):
    if prim_path in added_objects:
        del added_objects[prim_path]
        commands.execute("DeletePrimsCommand", paths=[prim_path])
    else:
        carb.log_warn(f"Tried to Remove, but Object not found: {prim_path}")

def sample_random_item_position(item_bbox_range: Tuple[List[float], List[float], List[float]], 
                                box_size: npt.NDArray[np.float_] = BOX_SIZE_INNER,
                                z_offset: float = 0.0) -> Gf.Vec3d:
    assert len(box_size) == 3
    box_x, box_y, box_z = box_size
    item_x_range, item_y_range, item_z_range = item_bbox_range
    place_x_range = [-box_x/2 - item_x_range[0], box_x/2 - item_x_range[1]]
    place_y_range = [-box_y/2 - item_y_range[0], box_y/2 - item_y_range[1]]
    print(f"Place X Range: {place_x_range}, Place Y Range: {place_y_range}")
    return Gf.Vec3d(np.random.uniform(*place_x_range),
                    np.random.uniform(*place_y_range),
                    BOX_SIZE_OUTER[2] - item_z_range[0] + z_offset)

def get_object_name_randomly() -> str:
    """
        Get object name randomly from the YCB dataset
    
    Returns:
        str: Object name (ex. "002_master_chef_can")
    """
    datas = glob.glob(os.path.join(OBJECTS_PATH, "*"))
    data = random.choice(datas)
    data_name = os.path.split(data)[1]
    return data_name


###################### Main Script ######################

simulation_world: World = World(stage_units_in_meters=1.0, physics_dt=1/480.0, rendering_dt=1/480.0)
# simulation_world: World = World(stage_units_in_meters=1.0)
# simulation_world.add_physics_callback("physics_callback",
#                                       lambda step_size: print("simulate with step: ", step_size))
# simulation_world.add_render_callback("render_callback",
#                                       lambda event: print("update app with step: ", event.payload["dt"]))
timeline = omni.timeline.get_timeline_interface()
# timeline.set_play_every_frame(False)  # 毎ステップsimulationのタイムステップが進むかどうか


# key: prim_path (str), value: ObjectInfo
added_objects: Dict[str, ItemInfo] = {}

# World setup
simulation_world.get_physics_context().enable_gpu_dynamics(True)
simulation_world.scene.add_default_ground_plane(z_position=0.0)

# Add Target Box
box_data = add_storage(Gf.Vec3d(0, 0, 0))

# Create lights
dome_light = prims.create_prim("/World/DomeLight", "DomeLight")
dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(3200.0)

# Wait for things to load
simulation_app.update()
while stage.is_stage_loading():
    simulation_app.update()

# need to initialize physics getting any articulation..etc
simulation_world.initialize_physics()
simulation_world.play()

import time
# import rclpy
# from geometry_msgs.msg import Pose
# from omni.isaac.dynamic_control import _dynamic_control

# node = rclpy.create_node("isaac_node")
# pose_publisher = node.create_publisher(Pose, "/pose", 10)
# dc = _dynamic_control.acquire_dynamic_control_interface()

import time
start = time.time()
MAX_OBJECTS = 40
GENERATE_INTERVAL = 500
counter = 0
while simulation_app.is_running():
    if counter % GENERATE_INTERVAL == 0 and counter // GENERATE_INTERVAL < MAX_OBJECTS:
        # object_name = "002_master_chef_can" # Get Randomly
        object_name = get_object_name_randomly()
        prim_name = "ycb_" + object_name + "_" + str(counter // GENERATE_INTERVAL)
        item_info = add_ycb_object(object_name, prim_name)
        position = sample_random_item_position(item_info.get_bbox_range(), z_offset=0.1)
        print(f"Moving object to: {position}")
        item_info.move_to(position=position)
    counter += 1
    simulation_world.step(render=True)
    if simulation_world.is_playing():
        pass
    end = time.time()
    # print("Actual FPS: ", 1/(end-start))
    start = end

simulation_world.step(render=True)


# Post processing
# Remove all objects
# for prim_path in list(added_objects.keys()):
#     remove_object(prim_path)

simulation_world.stop()
simulation_app.close()


