
import trimesh
import trimesh.bounds
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.spatial.transform import Rotation as R

HOME_PATH = os.environ["HOME"]
names = ["003_cracker_box", "011_banana"]
for name in names:
  mesh_path = os.path.join(HOME_PATH, f"research_ws/isaac_sim_ws/src/isaac_env/objects/ycb/{name}/google_16k/textured.obj")

  mesh: trimesh.Trimesh = trimesh.load(mesh_path)

  # 比較用描画
  xy_data = mesh.vertices[:, :2]
  bb_xy_data = mesh.bounding_box.vertices[:, :2]
  plt.figure()
  plt.plot(xy_data[:, 0], xy_data[:, 1], 'o', label='Data Points')
  plt.plot(bb_xy_data[:, 0], bb_xy_data[:, 1], 'o', label='AABB Data Points')
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.title(f'Data Scatter Plot before Transformation: {name}')
  plt.axis('equal')
  plt.legend()
  plt.grid(True)

  plt.show(block=False)


  # 提供されたデータのZ方向がわかるように、2DのOBB情報のみを使って回転変換
  to_origin_2d, extents_2d = trimesh.bounds.oriented_bounds_2D(mesh.vertices[:, :2])
  print(extents_2d)

  # 同次変換の適用と、重心位置への平行移動
  transform = np.eye(4)
  transform[:2, :2] = to_origin_2d[:2, :2]
  transform[:2, 3] = to_origin_2d[:2, 2]
  mesh.apply_transform(transform)
  center = mesh.centroid
  mesh.apply_translation(-center)

  xy_data = mesh.vertices[:, :2]
  bb_xy_data = mesh.bounding_box.vertices[:, :2]

  plt.figure()
  plt.plot(xy_data[:, 0], xy_data[:, 1], 'o', label='Data Points')
  plt.plot(bb_xy_data[:, 0], bb_xy_data[:, 1], 'o', label='AABB Data Points')

  plt.xlabel('X')
  plt.ylabel('Y')
  plt.title(f'Data Scatter Plot after Transformation: {name}')
  plt.axis('equal')
  plt.legend()
  plt.grid(True)

  plt.show(block=False)

plt.show()

# mesh.export(os.path.join(HOME_PATH, "research_ws/isaac_sim_ws/src/isaac_env/objects/ycb/002_master_chef_can/google_16k/textured_centered.obj"))

