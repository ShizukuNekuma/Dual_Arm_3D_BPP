import pymeshlab
import os

# メッシュとテクスチャを読み込む
ms = pymeshlab.MeshSet()
HOME_PATH = os.environ["HOME"]
name = "003_cracker_box"
mesh_path = os.path.join(HOME_PATH, f"research_ws/isaac_sim_ws/src/isaac_env/objects/ycb/{name}/google_16k/textured.obj")


ms.load_new_mesh(mesh_path)

# メッシュのテクスチャ情報を維持しながら簡略化
ms.meshing_decimation_quadric_edge_collapse_with_texture(
    targetfacenum=ms.current_mesh().face_number() // 4,
    qualitythr=0.5,
    preserveboundary=True,
    preservenormal=True,
    planarquadric=True,
)

# 簡略化後のメッシュを保存
ms.save_current_mesh('simplified_mesh.obj')

ms.show_polyscope()
