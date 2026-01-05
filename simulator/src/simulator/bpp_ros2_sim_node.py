import json
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import random
import webcolors

from simulator.bpp_action_parser import ActionType, EpisodeSettings, SettingAction, InitializeAction, GenerateAction, Vector3, parse_action

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid
from isaacsim.core.utils import prims, stage
from isaacsim.core.utils.extensions import enable_extension
from pxr import Usd, UsdGeom, Sdf, Gf

# enable ROS2 bridge extension
enable_extension("isaacsim.ros2.bridge")

simulation_app.update()

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from std_msgs.msg import String

###################### Constants ######################
STAGE_THICKNESS = 0.1
MAX_ITEM_SIZE = 0.5 # 500mm
CONTAINER_SIZE = Vector3([MAX_ITEM_SIZE*2, MAX_ITEM_SIZE*2, STAGE_THICKNESS])
CONTAINER_AREA_CENTER = Vector3([MAX_ITEM_SIZE, MAX_ITEM_SIZE, STAGE_THICKNESS/2])
BUFFER_AREA_CENTER = Vector3([CONTAINER_AREA_CENTER.x - MAX_ITEM_SIZE*2, CONTAINER_AREA_CENTER.y, STAGE_THICKNESS/2])
TEMPORARY_SAVE_AREA_CENTER = Vector3([CONTAINER_AREA_CENTER.x + MAX_ITEM_SIZE*2, CONTAINER_AREA_CENTER.y, STAGE_THICKNESS/2])
SIZE_SHRINK = np.array([0.001, 0.001, 0.0])
# DROP_OFFSET = np.array([0, 0, STAGE_THICKNESS + 0.01])
DROP_OFFSET = np.array([0, 0, STAGE_THICKNESS])

LIFT_HEIGHT = 1.5  # 提升高度
MOVE_SPEED = 2.0   # 虚拟臂移动速度 (m/s)

###################### Custom Classes ######################
@dataclass
class ItemInfo:
  cube: VisualCuboid
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

###################### ROS2 Node and Task Scheduler ######################
from collections import defaultdict

@dataclass
class ActionInfo:
    label: str
    size: Tuple[int, int, int]
    position: Tuple[int, int, int] | int # Can be (lx, ly, lz) for bin, or index for buffer/tbs

class ActionDependencyManager:
    def __init__(self):
        self.lock = threading.Lock()

        self._is_loaded = False
        # label -> "label", "size", "position"
        self.actions_info: Dict[str, ActionInfo] = {}
        # label -> list of actions that depend on this action
        self.results: Dict[str, List[str]] = defaultdict(list)
        # label -> number of remaining unfinished prerequisites
        self.indegree: Dict[str, int] = {}
        
        self.ready_actions = set()
        self.running_actions = set()
        self.finished_actions = set()

    def load_actions(self, action_list: List[dict]):
        with self.lock:
            # clear previous data
            self.actions_info.clear()
            self.results.clear()
            self.indegree.clear()
            self.ready_actions.clear()
            self.running_actions.clear()
            self.finished_actions.clear()

            for action in action_list:

                label = action["label"]
                self.actions_info[label] = ActionInfo(
                    label = label,
                    size = action["size"],
                    position = action["position"],
                )
                
                self.results[label] = action["results"]
                
                degree = len(action["preconditions"])
                self.indegree[label] = degree
                
                if degree == 0:
                    self.ready_actions.add(label)

        self._is_loaded = True

    def get_next_task(self, next_action) -> Optional[ActionInfo]:
        """ 
        The executor calls this function to get a task.

        Args:
            next_action: "pick" or "place_{i}"

        Returns:
            ActionInfo object, or None if no task is ready.
        """
        with self.lock:
            if next_action == "pick":
                for label in self.ready_actions:
                    if "pick" in label:
                        self.ready_actions.remove(label)
                        self.running_actions.add(label)
                        return self.actions_info[label]
            else:  # place
                for label in self.ready_actions:
                    if label == next_action:
                        self.ready_actions.remove(label)
                        self.running_actions.add(label)
                        return self.actions_info[label]
            # no ready task found
            return None

    def report_task_finished(self, label: str):
        """
        The executor calls this function to report task completion.
        New ready tasks may be generated.
        """
        with self.lock:
            if label not in self.running_actions:
                return
            
            self.running_actions.remove(label)
            self.finished_actions.add(label)
            
            for child_label in self.results[label]:
                if child_label in self.indegree:
                    self.indegree[child_label] -= 1
                    
                    if self.indegree[child_label] == 0:
                        self.ready_actions.add(child_label)
                        
    def is_all_done(self):
        with self.lock:
            flag = self._is_loaded and len(self.finished_actions) == len(self.actions_info)

            if flag == True:
                # clear previous data
                self._is_loaded = False
                self.actions_info.clear()
                self.results.clear()
                self.indegree.clear()
                self.ready_actions.clear()
                self.running_actions.clear()
                self.finished_actions.clear()
            
            return flag


class UpdateSceneSubscriber(Node):
    def __init__(self):
        super().__init__('update_scene_subscriber')

        self.updater_sub = self.create_subscription(
            String,
            '/bpp/update_scene',
            self.listener_callback,
            10)
        self.updater_sub  # prevent unused variable warning

        self.scheduler_sub = self.create_subscription(
            String,
            '/bpp/action_list',
            self.handle_request,
            10)
        self.scheduler_sub  # prevent unused variable warning

        self.scheduler_pub = self.create_publisher(String, '/bpp/mission_complete', 10)
        
        self.task_scheduler = ActionDependencyManager()
        self.arms = [
            VirtualArm(arm_id=0, manager=self.task_scheduler, node=self),
            VirtualArm(arm_id=1, manager=self.task_scheduler, node=self)
        ]

        # setting up the world  
        self.timeline = omni.timeline.get_timeline_interface()
        self.simulation_world: World = World(stage_units_in_meters=1.0, physics_dt=1/100.0, rendering_dt=1/100.0)
        self.simulation_world.scene.add_default_ground_plane(z_position=0.0)

        self.settings: Optional[EpisodeSettings] = None
        self.boxes_data = BoxesDataStorage()
        self.container_prim = self.add_container(position=CONTAINER_AREA_CENTER, size=CONTAINER_SIZE)
        self.buffer_prim = self.add_buffer_area(position=BUFFER_AREA_CENTER,
                                    size=np.array([MAX_ITEM_SIZE, MAX_ITEM_SIZE, STAGE_THICKNESS]))
        self.temporary_save_prim = self.add_temporary_save_area(position=TEMPORARY_SAVE_AREA_CENTER,
                                    size=np.array([MAX_ITEM_SIZE, MAX_ITEM_SIZE, STAGE_THICKNESS]))
        
        # Create lights
        dome_light = prims.create_prim("/World/DomeLight", "DomeLight")
        dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(1000.0)

        self.simulation_world.reset()

    def listener_callback(self, msg):
        self.get_logger().info(f"Received action: {msg.data}")
        item = parse_action(msg.data)
        action_type = item.action_type
        action = item.action

        ## Setting
        if action_type == ActionType.SETTING:
            assert isinstance(action, SettingAction)
            self.get_logger().info(f"Updating settings: buffer_size={action.settings.buffer_size}, temporary_save_size={action.settings.temporary_save_size}")
            self.settings = action.settings

        ## Initialize
        if action_type == ActionType.INITIALIZE:
            assert isinstance(action, InitializeAction)
            self.initialize_environment(action=action)

        ## Generate
        if action_type == ActionType.GENERATE:
            assert isinstance(action, GenerateAction)
            self.generate_item(action=action)
        
        pass

    def handle_request(self, msg):
        self.get_logger().info(f"Received action_list: {msg.data}")
        action_list = json.loads(msg.data)
        self.task_scheduler.load_actions(action_list)

    def add_fixed_box(self, name: str, position: np.ndarray, size: np.ndarray,
                    color: Union[str, np.ndarray]) -> FixedCuboid:
        prim_path=f"/World/{name}"
        if isinstance(color, str):
            color = np.array(list(webcolors.name_to_rgb(color)))

        fixed_box = FixedCuboid(
            prim_path=prim_path, name=name, position=position, scale=size, size=1.0, color=color
            )
        self.simulation_world.scene.add(fixed_box)
        return fixed_box

    def add_container(self, position: np.ndarray, size: np.ndarray) -> FixedCuboid:
        return self.add_fixed_box("container", position, size, "black")

    def add_buffer_area(self, position: np.ndarray, size: np.ndarray) -> FixedCuboid:
        return self.add_fixed_box("buffer_area", position, size, "blue")

    def add_temporary_save_area(self, position: np.ndarray, size: np.ndarray) -> FixedCuboid:
        return self.add_fixed_box("temporary_save_area", position, size, "green")

    def add_box(self, prim_path: str, name: str, position: np.ndarray, size: np.ndarray,
                color: Union[str, np.ndarray]) -> VisualCuboid:
        if isinstance(color, str):
            color = np.array(list(webcolors.name_to_rgb(color)))

        cube = VisualCuboid(
            prim_path=prim_path, name=name, position=position, scale=size, size=1.0, color=color
            )
        cube.initialize()
        print(f"{prim_path} is added")
        return cube

    def initialize_environment(self, action: InitializeAction) -> None:
        buffer_size, save_size = self.settings.buffer_size, self.settings.temporary_save_size
        ## initialize buffer area
        boxes = action.buffer_boxes
        self.buffer_prim.set_local_scale([MAX_ITEM_SIZE, MAX_ITEM_SIZE*buffer_size, STAGE_THICKNESS])
        for i, box in enumerate(boxes):
            size = box * 0.1
            index_position, index = self.get_position_from_index(i)
            position = index_position + np.array([0, 0, size.z/2])
            cube = self.add_box(prim_path=f"/World/box_{i}",
                                    name=f"box_{i}",
                                    position=position + DROP_OFFSET,
                                    size=size - SIZE_SHRINK,
                                    color=np.array([random.random(), random.random(), random.random()]))
            self.boxes_data.buffer[index] = ItemInfo(cube=cube, size=size, position=position)
    
        ## initialize temporary save area
        self.temporary_save_prim.set_local_scale([MAX_ITEM_SIZE, MAX_ITEM_SIZE*save_size, STAGE_THICKNESS])
        for i in range(save_size):
            self.boxes_data.temporary_save[i] = None

    def generate_item(self, action: GenerateAction) -> None:
        size = action.size * 0.1
        index_position, index = self.get_position_from_index(action.at)
        position = index_position + np.array([0, 0, size.z/2])
        cube = self.add_box(prim_path=f"/World/box_{self.boxes_data.num_total}",
                                name=f"box_{self.boxes_data.num_total}",
                                position=position + DROP_OFFSET,
                                size=size - SIZE_SHRINK,
                                color=np.array([random.random(), random.random(), random.random()]))
        self.boxes_data.buffer[index] = ItemInfo(cube=cube, size=size, position=position)

    def get_position_from_index(self, index: int) -> Tuple[np.ndarray, int]: # for place action
        """
            index情報から、対応するouter area上面の中心座標を取得する
        """
        buffer_size, save_size = self.settings.buffer_size, self.settings.temporary_save_size
        buffer_center = Vector3(self.buffer_prim.get_world_pose()[0])
        save_center = Vector3(self.temporary_save_prim.get_world_pose()[0])
        if index < buffer_size: # buffer area
            buffer_index = index
            y_start = - MAX_ITEM_SIZE * buffer_size / 2
            y = y_start + MAX_ITEM_SIZE * (buffer_index + 0.5)
            return buffer_center + np.array([0, y, 0]), buffer_index # , "buffer"
        if index < buffer_size + save_size: # temporary save area
            save_index = index - buffer_size
            y_start = - MAX_ITEM_SIZE * save_size / 2
            y = y_start + MAX_ITEM_SIZE * (save_index + 0.5)
            return save_center + np.array([0, y, 0]), save_index # , "save"

        raise ValueError(f"Invalid index: {index}")

    def pop_item_info_from_index(self, index: int) -> ItemInfo: # for pick action
        if index < self.settings.buffer_size:
            item = self.boxes_data.buffer[index]
            self.boxes_data.buffer[index] = None
            return item
        if index < self.settings.buffer_size + self.settings.temporary_save_size:
            item = self.boxes_data.temporary_save[index - self.settings.buffer_size]
            self.boxes_data.temporary_save[index - self.settings.buffer_size] = None
            return item
        raise ValueError(f"Invalid index: {index}")

    def find_and_pop_item_from_container(self, position: np.ndarray, size: np.ndarray) -> Optional[ItemInfo]: # for pick action
        """
        该函数仅用于 pick from container, 也就是 removal 的情况
        关于旋转，假定 remove 的时候不会有旋转操作
        """
        for i, item_info in enumerate(self.boxes_data.container):
            if np.allclose(position, item_info.position) and np.allclose(size, item_info.size):
                self.boxes_data.container.pop(i)
                return item_info
        return None

    def run_simulation(self):
        self.timeline.play()

        while simulation_app.is_running():
            self.simulation_world.step(render=True)
            rclpy.spin_once(self, timeout_sec=0.0)

            for arm in self.arms:
                arm.update()

            if self.task_scheduler.is_all_done():
                msg = String()
                msg.data = "True"
                self.scheduler_pub.publish(msg)

        # Cleanup
        self.timeline.stop()
        self.destroy_node()
        simulation_app.close()

###################### Virtual Robot Arm ######################
# TODO: ATTENTION!!!
# 机械臂的idle不是真正的idle，因为pick之后只能执行place，place之后只能执行pick
# 机械臂的动作序列是 pick -> place -> pick -> place ...

from enum import Enum, auto

class State(Enum):
  IDLE = auto()
  EXECUTING = auto()
#   LIFTING = auto()
#   MOVING = auto()
#   LOWERING = auto()

class VirtualArm:
    def __init__(self, arm_id: int, manager: ActionDependencyManager, node: UpdateSceneSubscriber):
        self.arm_id = arm_id
        self.manager = manager
        self.node = node
        
        self.current_task: ActionInfo = None
        self.state = State.IDLE
        
        # internal state variables
        self.target_item: ItemInfo = None
        self.target_pos: np.ndarray = None
        self.target_ori: np.ndarray = None # use size to represent orientation
        self.path_points = []

    def update(self):
        # update called every frame, driven by main simulation loop
        if self.state == State.IDLE:
            # hoding box
            # if self.target_item is not None:
            #     self.target_item.cube.set_world_pose(np.zeros(3))
            #     if not np.array_equal(self.target_ori, self.target_item.size):
            #         # rotate 90 degrees TODO
            #         self.target_item.cube.set_world_pose(orientation=[0.0, 0.0, 0.70710678, 0.70710678])

            # attempt to get a new task
            if self.current_task is None:
                next_action = "pick"
            else:
                next_action = "place_" + str(self.current_task.label.split("_")[1])
            task = self.manager.get_next_task(next_action)

            if task:
                self.start_sequence(task)
        
        elif self.state == State.EXECUTING:
            self.execute_step()

    def start_sequence(self, task: ActionInfo):
        # label examples: "pick_0", "place_0"
        is_pick = "pick" in task.label
        self.target_ori = np.array(task.size) * 0.1 # orientation encoded in size
        
        # pick action
        if is_pick:
            # get item info from buffer/tbs/container
            if isinstance(task.position, int):
                self.target_item = self.node.pop_item_info_from_index(task.position)
            else:
                size = np.array(task.size) * 0.1
                pos = np.array(task.position) * 0.1 + size/2 + DROP_OFFSET
                self.target_item = self.node.find_and_pop_item_from_container(pos, size)
            
            if self.target_item == None:
                print(task)
                print(self.node.boxes_data.buffer)
                print(self.node.boxes_data.temporary_save)
                print(self.node.boxes_data.container)

            # lift the item up
            curr_pos, _ = self.target_item.cube.get_world_pose()
            dest_pos = np.array([curr_pos[0], curr_pos[1], LIFT_HEIGHT])
            self.path_points = [dest_pos]
        # place action
        else:
            # get place position
            if isinstance(task.position, int):
                final_pos, _ = self.node.get_position_from_index(task.position)
                final_pos[2] += (self.target_item.size[2] / 2 + DROP_OFFSET[2]/2)
            else:
                size = np.array(task.size) * 0.1
                offset = np.array(task.position) * 0.1
                
                container_base = CONTAINER_AREA_CENTER - CONTAINER_SIZE/2
                final_pos = container_base + offset + (size / 2) + DROP_OFFSET
            
            waypoint_air = np.array([final_pos[0], final_pos[1], LIFT_HEIGHT])
            self.path_points = [waypoint_air, final_pos]

        self.current_task = task
        self.target_pos = self.path_points.pop(0)
        self.state = State.EXECUTING
        self.node.get_logger().info(f"Arm {self.arm_id} started {task.label}")

    def execute_step(self):
        curr_pos, _ = self.target_item.cube.get_world_pose()
        direction = self.target_pos - curr_pos
        distance = np.linalg.norm(direction)
        
        step_size = MOVE_SPEED * (1/100.0) # 100Hz
        
        if distance > step_size:
            self.target_item.cube.set_world_pose(curr_pos + step_size * (direction / distance))
        else:
            self.target_item.cube.set_world_pose(position=self.target_pos)
            if not np.array_equal(self.target_ori, self.target_item.size):
                # rotate 90 degrees TODO
                self.target_item.cube.set_world_pose(orientation=[0.70710678, 0.0, 0.0, 0.70710678])

            if self.path_points:
                self.target_pos = self.path_points.pop(0)
            else:
                self.complete_task()

    def complete_task(self):
        # report completion to update dependencies
        self.manager.report_task_finished(self.current_task.label)

        if "place" in self.current_task.label:
            # update storage data
            self.target_item.position = self.target_pos
            self.target_item.size = self.target_ori
            if isinstance(self.current_task.position, int):
                # place in tbs
                self.node.boxes_data.temporary_save[self.current_task.position - self.node.settings.buffer_size] = self.target_item
            else:
                # place in container
                self.node.boxes_data.container.append(self.target_item)
            
            # cleanup
            self.current_task = None
            self.target_item = None

        # ready for next task
        self.state = State.IDLE
        self.target_ori = None
        self.target_pos = None


###################### Main Script ######################
if __name__ == "__main__":
    rclpy.init()
    node = UpdateSceneSubscriber()
    node.run_simulation()
    rclpy.shutdown()