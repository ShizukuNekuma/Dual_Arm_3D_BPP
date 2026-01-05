from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List
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
  SETTING = auto()
  INITIALIZE = auto()
  GENERATE = auto()


@dataclass
class EpisodeSettings:
  buffer_size: int
  temporary_save_size: int

@dataclass
class ActionBase:

  @property
  def type(self):
    return NotImplementedError("This is an abstract class")

@dataclass
class SettingAction(ActionBase):
  settings: EpisodeSettings

  @property
  def type(self):
    return ActionType.SETTING
  
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

Action = SettingAction | InitializeAction | GenerateAction


@dataclass
class EpisodeItem:
  action_type: ActionType | None = None
  action: Action | None = None

  def __post_init__(self):
    if self.action_type is None:
      self.action_type = self.action.type
    if self.action is None:
      raise ValueError("action is required")


def deserializer(obj: Dict[Any, Any]):
  # 自動的にやりたい インスタンスのattributeを見て、型を見て、適切な型に変換する
  if "buffer_boxes" in obj:
    obj["buffer_boxes"] = [Vector3(box) for box in obj["buffer_boxes"]]
  
  if "size" in obj:
    obj["size"] = Vector3(obj["size"])
    
  if "action_type" in obj and 'action' in obj:
    action_type = ActionType[obj["action_type"]]
    if action_type == ActionType.SETTING:
      return EpisodeItem(action_type, SettingAction(EpisodeSettings(**obj["action"])))
    if action_type == ActionType.INITIALIZE:
      return EpisodeItem(action_type, InitializeAction(**obj["action"]))
    if action_type == ActionType.GENERATE:
      return EpisodeItem(action_type, GenerateAction(**obj["action"]))

  return obj

def parse_action(json_str):
  return json.loads(json_str, object_hook=deserializer)