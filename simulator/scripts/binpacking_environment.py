from isaacsim import SimulationApp

from test.load_trajectories import ActionType, EpisodeRecorder, EpisodeSettings, GenerateAction, InitializeAction, PackingAction, RemovalAction, Vector3

simulation_app = SimulationApp({"headless": False})

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union, List

import carb
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim 
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils import prims, stage

from pxr import Usd, UsdGeom, Sdf, Gf
import webcolors

###################### Constants ######################
STAGE_THICKNESS = 0.1
MAX_ITEM_SIZE = 0.5 # 500mm
CONTAINER_SIZE = Vector3([MAX_ITEM_SIZE*2, MAX_ITEM_SIZE*2, STAGE_THICKNESS])
CONTAINER_AREA_CENTER = Vector3([MAX_ITEM_SIZE, MAX_ITEM_SIZE, STAGE_THICKNESS/2])
BUFFER_AREA_CENTER = Vector3([CONTAINER_AREA_CENTER.x - 1, CONTAINER_AREA_CENTER.y, STAGE_THICKNESS/2])
TEMPORARY_SAVE_AREA_CENTER = Vector3([CONTAINER_AREA_CENTER.x + 1, CONTAINER_AREA_CENTER.y, STAGE_THICKNESS/2])
SIZE_SHRINK = np.array([0.001, 0.001, 0.0])
DROP_OFFSET = np.array([0, 0, STAGE_THICKNESS + 0.01])

###################### Custom Classes ######################

@dataclass
class ItemInfo:
  xform_prim: SingleXFormPrim
  size: np.ndarray
  position: np.ndarray

@dataclass
class BoxesDataStorage:
  buffer: Dict[int, Optional[ItemInfo]] = field(default_factory=dict) 
  temporary_save: Dict[int, Optional[ItemInfo]] = field(default_factory=dict)
  container: List[ItemInfo] = field(default_factory=list)

  @property
  def num_buffer(self) -> int:
    return len([v for v in self.buffer.values() if v is not None])
  
  @property
  def num_temporary_save(self) -> int:
    return len([v for v in self.temporary_save.values() if v is not None])
  
  @property
  def num_container(self) -> int:
    return len(self.container)
  
  @property
  def num_total(self) -> int:
    return self.num_buffer + self.num_temporary_save + self.num_container

###################### Functions ######################
def add_fixed_box(world: World, name: str, position: np.ndarray, size: np.ndarray,
                  color: Union[str, np.ndarray]) -> SingleXFormPrim:
  prim_path=f"/World/{name}"
  if isinstance(color, str):
    color = np.array(list(webcolors.name_to_rgb(color)))
  xform_prim = world.scene.add(
    FixedCuboid(
      prim_path=prim_path, name=name, position=position, scale=size, size=1.0, color=color
    )
  )
  return xform_prim

def add_container(world: World, position: np.ndarray, size: np.ndarray) -> SingleXFormPrim:
  return add_fixed_box(world, "container", position, size, "black")

def add_buffer_area(world: World, position: np.ndarray, size: np.ndarray) -> SingleXFormPrim:
  return add_fixed_box(world, "buffer_area", position, size, "blue")

def add_temporary_save_area(world: World, position: np.ndarray, size: np.ndarray) -> SingleXFormPrim:
  return add_fixed_box(world, "temporary_save_area", position, size, "green")

def add_box(world: World, prim_path: str, name: str, position: np.ndarray, size: np.ndarray,
            color: Union[str, np.ndarray]) -> SingleXFormPrim:
  if isinstance(color, str):
    color = np.array(list(webcolors.name_to_rgb(color)))
  xform_prim = world.scene.add(
    DynamicCuboid(
      prim_path=prim_path, name=name, position=position, scale=size, size=1.0, color=color
    )
  )
  return xform_prim

def initialize_environment(settings: EpisodeSettings, action: InitializeAction,
                           buffer_prim: SingleXFormPrim, temporary_save_prim: SingleXFormPrim,
                           boxes_data: BoxesDataStorage) -> None:
  buffer_size, save_size = settings.buffer_size, settings.temporary_save_size
  ## initialize buffer area
  boxes = action.buffer_boxes
  buffer_prim.set_local_scale([MAX_ITEM_SIZE, MAX_ITEM_SIZE*buffer_size, STAGE_THICKNESS])
  for i, box in enumerate(boxes):
    size = box * 0.1
    index_position, index = get_position_from_index(i, settings, boxes_data, buffer_prim, temporary_save_prim)
    position = index_position + np.array([0, 0, size.z/2])
    xform_prim = add_box(simulation_world,
                        prim_path=f"/World/box_{i}",
                        name=f"box_{i}",
                        position=position + DROP_OFFSET,
                        size=size - SIZE_SHRINK,
                        color=np.array([random.random(), random.random(), random.random()]))
    boxes_data.buffer[index] = ItemInfo(xform_prim=xform_prim, size=size, position=position)
  
  ## initialize temporary save area
  temporary_save_prim.set_local_scale([MAX_ITEM_SIZE, MAX_ITEM_SIZE*save_size, STAGE_THICKNESS])
  for i in range(buffer_size, buffer_size + save_size):
    boxes_data.temporary_save[i] = None

def generate_item(settings: EpisodeSettings, action: GenerateAction, 
                  buffer_prim: SingleXFormPrim, temporary_save_prim: SingleXFormPrim,
                  boxes_data: BoxesDataStorage) -> None:
  size = action.size * 0.1
  position, index = get_position_from_index(action.at, settings, boxes_data, buffer_prim, temporary_save_prim)
  position = position + np.array([0, 0, size.z/2])
  xform_prim = add_box(simulation_world,
                      prim_path=f"/World/box_{boxes_data.num_total}",
                      name=f"box_{boxes_data.num_total}",
                      position=position + DROP_OFFSET,
                      size=size - SIZE_SHRINK,
                      color=np.array([random.random(), random.random(), random.random()]))
  boxes_data.buffer[index] = ItemInfo(xform_prim=xform_prim, size=size, position=position)


def pack_item(settings: EpisodeSettings, action: PackingAction, boxes_data: BoxesDataStorage) -> None:
  item_info = pop_item_info_from_index(action.from_position, settings, boxes_data)
  # rescale item based on the size -> 将来的には回転操作いれる
  size = action.size * 0.1
  item_info.xform_prim.set_local_scale(size - SIZE_SHRINK)
  to_position = action.to_position * 0.1 + size / 2
  item_info.xform_prim.set_local_pose(translation=to_position + DROP_OFFSET, orientation=[1, 0, 0, 0])
  boxes_data.container.append(item_info)
  item_info.position = to_position
  item_info.size = size

def remove_item(settings: EpisodeSettings, action: RemovalAction, boxes_data: BoxesDataStorage,
                buffer_prim: SingleXFormPrim, temporary_save_prim: SingleXFormPrim) -> None:
  from_position = action.from_position * 0.1
  size = action.size * 0.1
  item_info = find_and_pop_item_from_container(from_position + size/2, size, boxes_data)
  if item_info is None:
    raise ValueError("Item not found in the container")
  to_position, index = get_position_from_index(action.to_position, settings, boxes_data, buffer_prim, temporary_save_prim)
  to_position = to_position + np.array([0, 0, size.z/2])
  item_info.xform_prim.set_local_pose(translation=to_position + DROP_OFFSET, orientation=[1, 0, 0, 0])
  item_info.position = to_position
  boxes_data.temporary_save[index] = item_info


def find_and_pop_item_from_container(position: np.ndarray, size: np.ndarray, boxes_data: BoxesDataStorage) -> Optional[ItemInfo]:
  ## TODO: サイズに関しては、回転を考慮する必要あり -> pack_itemでの回転量の取り扱い次第
  for i, item_info in enumerate(boxes_data.container):
    if np.allclose(position, item_info.position) and np.allclose(size, item_info.size):
      boxes_data.container.pop(i)
      return item_info
  return None

def pop_item_info_from_index(index: int, settings: EpisodeSettings, boxes_data: BoxesDataStorage) -> ItemInfo:
  if index < settings.buffer_size:
    item = boxes_data.buffer[index]
    boxes_data.buffer[index] = None
    return item
  if index < settings.buffer_size + settings.temporary_save_size:
    item = boxes_data.temporary_save[index - settings.buffer_size]
    boxes_data.temporary_save[index - settings.buffer_size] = None
    return item
  raise ValueError(f"Invalid index: {index}")

def get_position_from_index(index: int, settings: EpisodeSettings,
                            data_boxes: BoxesDataStorage,
                            buffer_prim: SingleXFormPrim, temporary_save_prim: SingleXFormPrim) -> Tuple[np.ndarray, int]:
  """
    index情報から、対応するouter area上面の中心座標を取得する
  """
  buffer_size, save_size = settings.buffer_size, settings.temporary_save_size
  buffer_center = Vector3(buffer_prim.get_world_pose()[0])
  save_center = Vector3(temporary_save_prim.get_world_pose()[0])
  if index < buffer_size:
    buffer_index = index
    y_start = - MAX_ITEM_SIZE * buffer_size / 2
    y = y_start + MAX_ITEM_SIZE * (buffer_index + 0.5)
    return buffer_center + np.array([0, y, 0]), buffer_index # , "buffer"
  elif index > buffer_size + save_size:
    # find first index of the save area which is None
    save_index: Optional[int] = None
    for i, item_info in enumerate(data_boxes.temporary_save.values()):
      if item_info is None:
        save_index = i
        break
    if save_index is None:
      raise ValueError("Temporary save area is full")
    y_start = - MAX_ITEM_SIZE * save_size / 2
    y = y_start + MAX_ITEM_SIZE * (save_index + 0.5)
    return save_center + np.array([0, y, 0]), save_index # , "save"
  raise ValueError(f"Invalid index: {index}")



###################### Main Script ######################
#TODO
# Resetting the world needs to be called before querying anything related to an articulation specifically.
# Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
# world.reset()

simulation_world: World = World(stage_units_in_meters=1.0, physics_dt=1/100.0, rendering_dt=1/100.0)

boxes_data = BoxesDataStorage()

# World setup
simulation_world.get_physics_context().enable_gpu_dynamics(True)
simulation_world.scene.add_default_ground_plane(z_position=0.0)

# Add Target Box
container_prim = add_container(simulation_world, position=CONTAINER_AREA_CENTER, size=CONTAINER_SIZE)
buffer_prim = add_buffer_area(simulation_world, position=BUFFER_AREA_CENTER,
                              size=np.array([MAX_ITEM_SIZE, MAX_ITEM_SIZE, STAGE_THICKNESS]))
temporary_save_prim = add_temporary_save_area(simulation_world,
                                              position=TEMPORARY_SAVE_AREA_CENTER,
                                              size=np.array([MAX_ITEM_SIZE, MAX_ITEM_SIZE, STAGE_THICKNESS]))

# Create lights
dome_light = prims.create_prim("/World/DomeLight", "DomeLight")
dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(1000.0)

# Wait for things to load
simulation_app.update()
while stage.is_stage_loading():
  simulation_app.update()

simulation_world.play()

import time
import rclpy
from geometry_msgs.msg import Pose

# node = rclpy.create_node("isaac_node")
# pose_publisher = node.create_publisher(Pose, "/pose", 10)
# dc = _dynamic_control.acquire_dynamic_control_interface()


recorder = EpisodeRecorder().load("logs/trajectories.json")
settings = recorder.settings
episode = recorder.episodes[str(0)]

import time
start = time.time()
counter = 0
GENERATE_INTERVAL = 200
while simulation_app.is_running():
  if counter % GENERATE_INTERVAL == 0 and len(episode) > 0:
    record = episode.pop(0)

    ## Initialize
    if record.action_type == ActionType.INITIALIZE:
      assert isinstance(record.action, InitializeAction)
      initialize_environment(settings, action=record.action, buffer_prim=buffer_prim,
                             temporary_save_prim=temporary_save_prim, boxes_data=boxes_data)

    ## Generate
    if record.action_type == ActionType.GENERATE:
      assert isinstance(record.action, GenerateAction)
      generate_item(settings, action=record.action, buffer_prim=buffer_prim,
                    temporary_save_prim=temporary_save_prim, boxes_data=boxes_data)
      pass

    ## Packing
    if record.action_type == ActionType.PACKING:
      assert isinstance(record.action, PackingAction)
      pack_item(settings, action=record.action, boxes_data=boxes_data)
      # size = record.action.size * 0.1
      # position = record.action.to_position * 0.1 + size/2
      # print(f"position: {position}, size: {size}")
      # index = counter // GENERATE_INTERVAL
      # item_info = add_box(simulation_world,
      #                     prim_path=f"/World/box_packing_{index}",
      #                     name=f"box_packing_{index}",
      #                     position=position + np.array([0, 0, STAGE_THICKNESS + 0.01]),
      #                     size=size - np.array([0.001, 0.001, 0]),
      #                     color=np.array([random.random(), random.random(), random.random()]))
    
    ## Removal
    if record.action_type == ActionType.REMOVAL:
      assert isinstance(record.action, RemovalAction)
      remove_item(settings, action=record.action, boxes_data=boxes_data,
                  buffer_prim=buffer_prim, temporary_save_prim=temporary_save_prim)


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