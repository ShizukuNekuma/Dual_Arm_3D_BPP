
from ast import Tuple
from dataclasses import dataclass, is_dataclass
from enum import Enum, auto
from tkinter import W
from typing import Any, Dict, List, Optional
import numpy as np
from numpy.typing import ArrayLike
import json


class Vector3(np.ndarray):
  def __new__(cls, value: ArrayLike):
    return np.asarray(value, dtype=float).view(cls)
  
  @property
  def x(self) -> float:
    return self[0]
  
  @property
  def y(self) -> float:
    return self[1]
  
  @property
  def z(self) -> float:
    return self[2]
  
  def __add__(self, other):
    return Vector3(super().__add__(other))

  def __mul__(self, other):
    return Vector3(super().__mul__(other))
  
  def __sub__(self, other):
    return Vector3(super().__sub__(other))
  
  def __truediv__(self, other):
    return Vector3(super().__truediv__(other))
  
  def __floordiv__(self, other):
    return Vector3(super().__floordiv__(other))


class AreaType(Enum):
  BUFFER = auto()
  TEMPORARY = auto()
  CONTAINER = auto()


class ActionType(Enum):
  INITIALIZE = auto()
  GENERATE = auto()
  PACKING = auto()
  REMOVAL = auto()
  FINALIZE = auto()

@dataclass
class ActionBase:

  @property
  def type(self):
    return NotImplementedError("This is an abstract class")

  
@dataclass
class InitializeAction(ActionBase):
  buffer_boxes: List[Vector3]

  @property
  def type(self):
    return ActionType.INITIALIZE


@dataclass
class GenerateAction(ActionBase):
  area: AreaType
  at: int
  size: Vector3 

  @property
  def type(self):
    return ActionType.GENERATE


@dataclass
class PackingAction(ActionBase):
  size: Vector3
  from_position: int # actionのindex -> outer areaのbox index
  to_position: Vector3

  @property
  def type(self):
    return ActionType.PACKING


@dataclass
class RemovalAction(ActionBase):
  size: Vector3
  from_position: Vector3
  to_position: int # actionのindex

  @property
  def type(self):
    return ActionType.REMOVAL


@dataclass
class FinalizeAction(ActionBase):

  @property
  def type(self):
    return ActionType.FINALIZE


Action = InitializeAction | GenerateAction | PackingAction | RemovalAction | FinalizeAction


@dataclass
class EpisodeItem:
  action_type: ActionType | None = None
  action: Action | None = None

  def __post_init__(self):
    if self.action_type is None:
      self.action_type = self.action.type
    if self.action is None:
      raise ValueError("action is required")


@dataclass
class EpisodeSettings:
  buffer_size: int
  temporary_save_size: int


Episode = List[EpisodeItem]


class EpisodeRecorder:

  def __init__(self, settings: Optional[EpisodeSettings] = None ):
    self.episodes: Dict[int, Episode] = {}
    self.settings = settings
    self.episode_index = 0
    self.episodes[self.episode_index] = []

  def record(self, action: Action):
    self.episodes[self.episode_index].append(EpisodeItem(action=action))
  
  def finalize_episode(self):
    if len(self.episodes[self.episode_index]) == 0: return 
    self.episodes[self.episode_index].append(EpisodeItem(action=FinalizeAction()))
    self.episode_index += 1
    self.episodes[self.episode_index] = []
  
  @staticmethod
  def serializer(obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if isinstance(obj, Enum):
      return obj.name
    if is_dataclass(obj):
      return obj.__dict__
    raise TypeError(f"{type(obj)} is not JSON serializable")
  
  @staticmethod
  def deserializer(obj: Dict[Any, Any]):
    # 自動的にやりたい インスタンスのattributeを見て、型を見て、適切な型に変換する
    if "buffer_boxes" in obj:
      obj["buffer_boxes"] = [Vector3(box) for box in obj["buffer_boxes"]]
    
    if "size" in obj:
      obj["size"] = Vector3(obj["size"])
    
    if "from_position" in obj:
      if isinstance(obj["from_position"], list):
        obj["from_position"] = Vector3(obj["from_position"])
    
    if "to_position" in obj:
      if isinstance(obj["to_position"], list):
        obj["to_position"] = Vector3(obj["to_position"])
      
    if "action_type" in obj and 'action' in obj:
      action_type = ActionType[obj["action_type"]]
      if action_type == ActionType.INITIALIZE:
        return EpisodeItem(action_type, InitializeAction(**obj["action"]))
      if action_type == ActionType.GENERATE:
        return EpisodeItem(action_type, GenerateAction(**obj["action"]))
      if action_type == ActionType.PACKING:
        return EpisodeItem(action_type, PackingAction(**obj["action"]))
      if action_type == ActionType.REMOVAL:
        return EpisodeItem(action_type, RemovalAction(**obj["action"]))
      if action_type == ActionType.FINALIZE:
        return EpisodeItem(action_type, FinalizeAction())
    return obj


  def to_json(self) -> str:
    data = {} 
    data["settings"] = self.settings.__dict__
    data["episodes"] = self.episodes
    return json.dumps(data, default=self.serializer, indent=2)

  def export(self, path: str):
    with open(path, "w") as f:
      data = {} 
      data["settings"] = self.settings.__dict__
      data["episodes"] = self.episodes
      json.dump(data, f, default=self.serializer, indent=2)
    
  
  @classmethod
  def load(cls, path: str) -> 'EpisodeRecorder':
    with open(path, "r") as f:
      data = json.load(f, object_hook=cls.deserializer)
      cls.settings = EpisodeSettings(**data["settings"])
      cls.episodes = data["episodes"]
      return cls


if __name__ == '__main__':
  EpisodeRecorder.load("logs/trajectories.json")

