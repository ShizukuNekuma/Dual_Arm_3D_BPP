import json
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import random
import webcolors
import os

from simulator.bpp_action_parser import ActionType, EpisodeSettings, SettingAction, InitializeAction, GenerateAction, Vector3, parse_action

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils import prims, stage, rotations
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import SurfaceGripper
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Usd, UsdGeom, Sdf, Gf

# enable ROS2 bridge extension
enable_extension("isaacsim.ros2.bridge")

simulation_app.update()

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

###################### Constants ######################
# all in meters
SCALER = 0.5
STAGE_THICKNESS = 0.1 * SCALER
MAX_ITEM_SIZE = 0.5 * SCALER
CONTAINER_SIZE = Vector3([MAX_ITEM_SIZE*2, MAX_ITEM_SIZE*2, STAGE_THICKNESS])
CONTAINER_AREA_CENTER = Vector3([MAX_ITEM_SIZE, MAX_ITEM_SIZE, STAGE_THICKNESS/2])
BUFFER_AREA_CENTER = Vector3([CONTAINER_AREA_CENTER.x - MAX_ITEM_SIZE*2, CONTAINER_AREA_CENTER.y, STAGE_THICKNESS/2])
TEMPORARY_SAVE_AREA_CENTER = Vector3([CONTAINER_AREA_CENTER.x + MAX_ITEM_SIZE*2, CONTAINER_AREA_CENTER.y, STAGE_THICKNESS/2])

SIZE_SHRINK = np.array([0.001, 0.001, 0.0])
STAGE_OFFSET = np.array([0, 0, STAGE_THICKNESS])
DROP_OFFSET = np.array([0, 0, STAGE_THICKNESS + 0.01])

ARM_MOUNT_HEIGHT = STAGE_THICKNESS
ARM_Y_OFFSET = 0.5
ARM_SPACE = ARM_Y_OFFSET + 0.5
ARM_HEIGHT = ARM_MOUNT_HEIGHT + 1.5

ARM_URDF_PATH = "ur10_robot.urdf"  # 请确保该URDF文件存在于合适的位置

###################### Custom Classes ######################
@dataclass
class ItemInfo:
  cube: DynamicCuboid
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

###################### Dual Arm System Manager ######################
###################### Dual Arm System ######################
class DualArmSystem:
    """
    管理双臂系统的类。
    吸取了 ur10_pick_up.py 的经验，使用 SingleManipulator + SurfaceGripper。
    """
    def __init__(self, world: World):
        self.world = world
        self.robots = {}
        self.configs = {
            "arm_0": {"pos": np.array([CONTAINER_AREA_CENTER.x, CONTAINER_AREA_CENTER.y - ARM_Y_OFFSET, 0.0]), "ori": np.array([1., 0., 0., 0.])},
            "arm_1": {"pos": np.array([CONTAINER_AREA_CENTER.x, CONTAINER_AREA_CENTER.y + ARM_Y_OFFSET, 0.0]), "ori": np.array([0., 0., 0., 1.])},
        }
        
    # 配置两台 UR10：arm_0 放在 -Y，arm_1 放在 +Y
    def setup_robot(self):
        for name, config in self.configs.items():
            prim_path = f"/World/{name}"
            assets_root_path = get_assets_root_path()
            asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
            gripper_usd = assets_root_path + "/Isaac/Robots/UR10/Props/short_gripper.usd"
            add_reference_to_stage(usd_path=gripper_usd, prim_path=f"{prim_path}/ee_link")

            gripper = SurfaceGripper(
                end_effector_prim_path=f"{prim_path}/ee_link", 
                translate=0.1611, 
                direction="x"
            )
            
            ur10 = self.world.scene.add(
                SingleManipulator(
                    prim_path=prim_path, 
                    name=name, 
                    end_effector_prim_path=f"{prim_path}/ee_link", 
                    gripper=gripper
                )
            )
            
            # 设置初始位姿
            ur10.set_joints_default_state(positions=np.array([-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, 0]))
            ur10.gripper.set_default_state(opened=True)

            self.robots[name] = ur10

###################### ROS2 Node and Task Scheduler ######################
class TaskScheduler(Node):
    def __init__(self):
        super().__init__('task_scheduler')

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
        
        # setting up the world  
        self.timeline = omni.timeline.get_timeline_interface()
        self.simulation_world: World = World(stage_units_in_meters=1.0, physics_dt=1/100.0, rendering_dt=1/100.0)
        self.simulation_world.scene.add_default_ground_plane(z_position=0.0)

        self.dual_arm = DualArmSystem(self.simulation_world)
        self.dual_arm.setup_robot()

        self.settings: Optional[EpisodeSettings] = None
        self.boxes_data = BoxesDataStorage()
        self.rotated_boxes: List[str] = []
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
        # collect operated boxes
        pick_box_list, place_position_list = self.collect_operated_boxes(action_list)
        self.generate_PDDL_file("output/problem.pddl", action_list, pick_box_list)
        self.export_scene_to_json("output/initial_scene.json", pick_box_list, place_position_list)

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
                color: Union[str, np.ndarray]) -> DynamicCuboid:
        if isinstance(color, str):
            color = np.array(list(webcolors.name_to_rgb(color)))

        cube = DynamicCuboid(
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
            size = box * 0.1 * SCALER
            index_position, index = self.get_position_from_index(i)
            position = index_position + np.array([0, 0, size.z/2])
            cube = self.add_box(prim_path=f"/World/box_{i}",
                                    name=f"box_{i}",
                                    position=position + DROP_OFFSET/2,
                                    size=size - SIZE_SHRINK,
                                    color=np.array([random.random(), random.random(), random.random()]))
            self.boxes_data.buffer[index] = ItemInfo(cube=cube, size=size, position=position)
    
        ## initialize temporary save area
        self.temporary_save_prim.set_local_scale([MAX_ITEM_SIZE, MAX_ITEM_SIZE*save_size, STAGE_THICKNESS])
        for i in range(save_size):
            self.boxes_data.temporary_save[i] = None

    def generate_item(self, action: GenerateAction) -> None:
        size = action.size * 0.1 * SCALER
        index_position, index = self.get_position_from_index(action.at)
        position = index_position + np.array([0, 0, size.z/2])
        cube = self.add_box(prim_path=f"/World/box_{self.boxes_data.num_total}",
                                name=f"box_{self.boxes_data.num_total}",
                                position=position + DROP_OFFSET/2,
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
    
    def collect_operated_boxes(self, action_list: List[dict]):
        """
        快速跑一遍action sequence, 确定每个pick操作的box在仿真环境中的名字, 并将其与pick的label绑定
        对于place操作, 则需要记录机械臂放手的位置, 并与place的label绑定
        返回两个列表, 存储的是pair, 格式为 (label, box_name) 和 (label, place_position)
        """
        pick_box_list = []
        place_position_list = []
        container_base = CONTAINER_AREA_CENTER - CONTAINER_SIZE/2

        for action in action_list:
            label = action["label"]
            position = action["position"]
            size = action["size"]
            if "pick" in label:
                target_ori = np.array(size) * 0.1 * SCALER
                if isinstance(position, int):
                    current_item = self.pop_item_info_from_index(position)
                    pick_box_list.append( (label, current_item.cube.name) )
                else:
                    size = np.array(size) * 0.1 * SCALER
                    offset = np.array(position) * 0.1 * SCALER
                    pos = container_base + offset + size/2 + STAGE_OFFSET
                    current_item = self.find_and_pop_item_from_container(pos, size)
                    pick_box_list.append( (label, current_item.cube.name) )
            elif "place" in label:
                if isinstance(position, int):
                    final_pos, _ = self.get_position_from_index(position)
                    final_pos[2] += (size[2] + STAGE_OFFSET[2]/2)
                    place_position_list.append( (label, final_pos.tolist()) )
                else:
                    size = np.array(size) * 0.1 * SCALER
                    offset = np.array(position) * 0.1 * SCALER
                    final_pos = container_base + offset + size/2 + STAGE_OFFSET
                    place_position_list.append( (label, final_pos.tolist()) )
                
                # 判断是否发生旋转
                if not np.array_equal(target_ori, current_item.size):
                    self.rotated_boxes.append(current_item.cube.name)
                # 更新 boxes_data
                # update storage data
                current_item.position = final_pos
                current_item.size = target_ori
                if isinstance(position, int):
                    # place in tbs
                    self.boxes_data.temporary_save[position - self.settings.buffer_size] = current_item
                else:
                    # place in container
                    self.boxes_data.container.append(current_item)

        return pick_box_list, place_position_list

    def generate_PDDL_file(self, filepath: str, action_list: List[dict], pick_box_list: List[Tuple[str, str]]):
        """
        生成PDDL问题文件, 包含初始状态和目标状态
        """

        pick_task_to_box = {item[0]: item[1] for item in pick_box_list}
        
        # 集合用于存储对象，避免重复
        tasks = []
        boxes = set()
        targets = []
        
        # 存储生成的 PDDL 语句列表
        precedes_stmts = []
        task_def_stmts = []
        init_ready_stmts = []
        goal_stmts = []

        # 遍历 action_list 处理逻辑
        for action in action_list:
            label = action['label']
            tasks.append(label)

            init_ready_stmts.append(f"(task-not-done {label})")
            
            # 1. 处理依赖关系 (Precedence)
            # 既然 results 表示后续节点，直接生成 precedes 关系
            for child_label in action.get('results', []):
                precedes_stmts.append(f"(precedes {label} {child_label})")
            

            # 2. 处理任务定义 (Task Definition) 和 Objects 提取
            if "pick" in label:
                # 处理 Pick 任务
                if label in pick_task_to_box:
                    box_name = pick_task_to_box[label]
                    boxes.add(box_name)
                    task_def_stmts.append(f"(is-pick-task {label} {box_name})")
                else:
                    print(f"Warning: {label} not found in pick_box_list")
                    
            elif "place" in label:
                # 处理 Place 任务
                # 假设命名规范为 place_X，对应的 pick 为 pick_X
                # 通过 split('_')[-1] 获取 ID
                suffix = label.split('_')[-1]
                target_name = f"target_{suffix}"
                targets.append(target_name)
                
                # 找到对应的 pick 任务以确定 box
                associated_pick = f"pick_{suffix}"
                box_name = pick_task_to_box.get(associated_pick)
                
                if box_name:
                    boxes.add(box_name) # 确保 box 被添加
                    task_def_stmts.append(f"(is-place-task {label} {box_name} {target_name})")
                    goal_stmts.append(f"(task-done {label})") # 所有 place 任务视为目标
                else:
                    print(f"Warning: Could not find box for {label} (looked for {associated_pick})")

        # ===========================
        # 3. 组装 Problem 字符串
        # ===========================
        
        # 格式化 Objects 部分
        # 排序以保持输出确定性
        manipulators_str = "flange_tool0 -manipulator"
        boxes_str = " ".join(sorted(list(boxes)))
        targets_str = " ".join(sorted(targets))
        tasks_str = " ".join(tasks)
        
        objects_block = f"""  (:objects
        {manipulators_str}
        {boxes_str} -objs
        {targets_str} -pos
        {tasks_str} -task)"""

        # 格式化 Init 部分
        init_block = "  (:init\n    (arm-empty flange_tool0)\n"
        init_block += "\n    " + "\n    ".join(init_ready_stmts)
        init_block += "\n\n    " + "\n    ".join(precedes_stmts)
        init_block += "\n\n    " + "\n    ".join(task_def_stmts)
        init_block += "\n  )"

        # 格式化 Goal 部分
        goal_block = "  (:goal (and\n    " + "\n    ".join(goal_stmts) + "\n  ))"

        problem_pddl = f"""(define (problem dual-arm-dag-task)
        (:domain dual-arm-scheduling)
        {objects_block}
        {init_block}
        {goal_block}
        )"""

        # ===========================
        # 4. 写入文件
        # ===========================
        
        with open(filepath, "w") as f:
            f.write(problem_pddl)
            
        self.get_logger().info(f"Successfully generated problem.pddl")

    def export_scene_to_json(self, filepath: str, pick_box_list: List[Tuple[str, str]], place_position_list: List[Tuple[str, float, float, float]]):
        """
        导出当前仿真环境状态为JSON文件
        """
        # extract picked box names from pick_box_list
        picked_box_names = set()
        for _, box_name in pick_box_list:
            picked_box_names.add(box_name)    

        def cuboid_to_dict(cuboid, color=None, is_surface=False):
            pos, ori = cuboid.get_world_pose()
            size = cuboid.get_local_scale()
            name = cuboid.name
            obj = {
                "name": name,
                "pose": {
                    "translation": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "rotation": [float(ori[1]), float(ori[2]), float(ori[3]), float(ori[0])] # modify quaternion order
                },
                "shape": {
                    "type": "box",
                    "extents": [float(size[0])/2, float(size[1])/2, float(size[2])/2]
                },
                "is_static": False if name in picked_box_names else True
            }
            if color is None:
                color = [random.random(), random.random(), random.random()]
            obj["color"] = color + [1.0]
            obj["specular"] = [0.4, 0.4, 0.8]
            if is_surface:
            # calculate surface z-axis value (relative)
                surface_z = float(size[2]) / 2.0
                obj["surface_bounds"] = {
                    "x": [-0.001, 0.001],
                    "y": [-0.001, 0.001],
                    "z": [surface_z, surface_z + 0.01]
                }
            return obj

        objects = []

        # FixedCuboid: container, buffer_area, temporary_save_area
        objects.append(cuboid_to_dict(self.container_prim, [0, 0, 0], is_surface=False))  # black
        objects.append(cuboid_to_dict(self.buffer_prim, [0, 0, 1], is_surface=False))   # blue
        objects.append(cuboid_to_dict(self.temporary_save_prim, [0, 1, 0], is_surface=False))  # green

        # DynamicCuboid: buffer, temporary_save, container
        for item in self.boxes_data.buffer.values():
            if item is not None:
                objects.append(cuboid_to_dict(item.cube, is_surface=True))
        for item in self.boxes_data.temporary_save.values():
            if item is not None:
                objects.append(cuboid_to_dict(item.cube, is_surface=True))
        for item in self.boxes_data.container:
            objects.append(cuboid_to_dict(item.cube, is_surface=True))

        # 添加机械臂放置box的位置，为了保证一致性，将放置位置作为虚拟物体加入objects
        for label, place_pos in place_position_list:
            # 从 pick_box_list 中找到对应的 box
            box_name = None
            for pick_label, box_name in pick_box_list:
                if pick_label.split('_')[-1] == label.split('_')[-1]:
                    break
            suffix = label.split('_')[-1]
            obj_name = f"target_{suffix}"
            # 通过先前建立的 objects 获取 box_name 对应的 box 大小，也就是 extents
            size = None
            for item in objects:
                if item["name"] == box_name:
                    size = item["shape"]["extents"]
                    break
            pos = np.array(place_pos)
            # 通过将虚拟物体放置在z=-1处隐藏物体
            offset = float(pos[2]) + 1.0
            obj = {
                "name": obj_name,
                "pose": {
                    "translation": [float(pos[0]), float(pos[1]), -1.0],
                    # 根据是否发生旋转调整rotation
                    # TODO 考虑重复操作同一物体的情况
                    "rotation": [0.0, 0.0, 0.0, 1.0] if box_name not in self.rotated_boxes else [0.0, 0.0, 0.70710678, 0.70710678]
                },
                "shape": {
                    "type": "box",
                    "extents": [float(size[0])/2, float(size[1])/2, float(size[2])/2]
                },
                "is_virtual": True,
                "surface_bounds": {
                    "x": [-0.001, 0.001],
                    "y": [-0.001, 0.001],
                    "z": [float(size[2]) / 2.0 + offset, float(size[2]) / 2.0 + offset + 0.01]
                }
            }
            objects.append(obj)

        # 自动估算 bounds
        # x轴：buffer区左侧到temporary_save区右侧
        min_x = BUFFER_AREA_CENTER.x - MAX_ITEM_SIZE/2
        max_x = TEMPORARY_SAVE_AREA_CENTER.x + MAX_ITEM_SIZE/2
        # y轴：以container为中心，覆盖三块区域的最大宽度
        y_center = CONTAINER_AREA_CENTER.y
        y_half = CONTAINER_SIZE.y / 2
        min_y = y_center - y_half - ARM_SPACE
        max_y = y_center + y_half + ARM_SPACE
        # z轴：地面到最大操作高度
        min_z = -1.5
        max_z = ARM_HEIGHT + MAX_ITEM_SIZE

        bounds = [
            [float(min_x), float(max_x)],
            [float(min_y), float(max_y)],
            [float(min_z), float(max_z)]
        ]

        # *** 关键：添加 Robots ***
        robots_json = []
        for arm_name in ["arm_0", "arm_1"]:
            # 获取Base位置
            # base_pos, base_rot = self.dual_arm.get_base_pose(arm_name)
            config = self.dual_arm.configs[arm_name]
            base_pos = config["pos"].tolist()
            
            # 获取当前关节角度
            joints = self.dual_arm.robots[arm_name].get_joint_positions()
            joint_map = {
                "shoulder_pan_joint": float(joints[0]),
                "shoulder_lift_joint": float(joints[1]),
                "elbow_joint": float(joints[2]),
                "wrist_1_joint": float(joints[3]),
                "wrist_2_joint": float(joints[4]),
                "wrist_3_joint": float(joints[5])
            }
            angle = 0.0 if arm_name == "arm_0" else 3.141592653589793
            robot_entry = {
                "name": arm_name,
                "urdf": ARM_URDF_PATH, # 这是一个指向外部文件的路径
                "is_fixed": True,
                "base_pose": {
                    "translation": [base_pos[0], base_pos[1]],
                    # Isaac Sim Quaternion is (w, x, y, z), JSON usually expects (x, y, z, w) or Euler
                    "rotation": angle
                },
                "joint_poses": joint_map,
            }
            robots_json.append(robot_entry)

        scene_dict = {
            "bounds": bounds,
            "robots": robots_json,
            "objects": objects
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(scene_dict, f, indent=1)
        self.get_logger().info(f"Scene exported to {filepath}")

    def run_simulation(self):
        self.timeline.play()
        i = 0
        while simulation_app.is_running():
            self.simulation_world.step(render=True)
            if i == 0:
                for name, config in self.dual_arm.configs.items():
                    self.dual_arm.robots[name].set_world_pose(position=config["pos"], orientation=config["ori"])
                i += 1
            if i == 1:
                for name, config in self.dual_arm.configs.items():
                    self.dual_arm.robots[name].set_joint_positions(positions=np.array([-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, 0]))
                i += 1
            rclpy.spin_once(self, timeout_sec=0.0)

            # if is_all_done():
            #     msg = String()
            #     msg.data = "True"
            #     self.scheduler_pub.publish(msg)

        # Cleanup
        self.timeline.stop()
        self.destroy_node()
        simulation_app.close()

###################### Main Script ######################
if __name__ == "__main__":
    rclpy.init()
    node = TaskScheduler()
    node.run_simulation()
    rclpy.shutdown()