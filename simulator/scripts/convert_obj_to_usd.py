
import argparse
import asyncio
import os
from typing import List

from isaacsim import SimulationApp

## Note:
# USDファイル生成時のtextureで参照しているpngファイルは、相対パスではなく絶対パスで保存されているので
# それぞれの環境でデータのconvertを実行する必要がある


async def convert(in_file, out_file, load_materials=True):
    # This import causes conflicts when global
    import omni.kit.asset_converter

    def progress_callback(progress, total_steps):
      print("convert progress_callback")
      pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    # setup converter and flags
    converter_context.ignore_materials = not load_materials
    # converter_context.single_mesh = True
    # converter_context.smooth_normals = True
    # converter_context.preview_surface = False
    ############# WARNING: USDファイルの単位はこれを設定しても変わってないかもしれない
    # Isaac SimのGUIでロードすると失敗することが多い
    # もし表示が変わらなければ、ここで Metric Assembler を無効化するか
    # ロード後に"unit mismatch found"といった情報が出るときは、transformのmeter per unitsが0.01になっているのを戻す必要がある
    converter_context.use_meter_as_world_unit = True # モデルがセンチメートルの場合、ワールド単位をメートルに設定
    # converter_context.create_world_as_default_root_prim = False
    converter_context.convert_fbx_to_z_up = True # Always use Z-up for fbx import.
    converter_context.keep_all_materials = True
    converter_context.export_hidden_props = True
    converter_context.bake_mdl_material = True

    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(in_file, out_file, progress_callback, converter_context)
    success = True
    while True:
        success = await task.wait_until_finished()
        if not success:
            await asyncio.sleep(0.1)
        else:
            break
    return success


def asset_convert(folders, max_models, load_materials):
    import omni.client
    import trimesh
    import trimesh.bounds
    import shutil
    import numpy as np

    supported_file_formats = ["obj"]

    for folder in folders:
        # データの前処理
        # meshデータの原点をbounding boxの中心となるように移動する
        print(f"\n\nTranslating Mesh Data in {folder}...\n")

        (result, models) = omni.client.list(folder)
        for i, entry in enumerate(models):

            model = str(entry.relative_path)
            model_name = os.path.splitext(model)[0] 
            model_format = (os.path.splitext(model)[1])[1:]
            if model_format in ["obj"]:
                input_mesh_model_path = folder + "/" + model
                output_mesh_model_path = folder + "/" + model_name + "_transformed.obj"

                # 変換対象かどうか整理
                if os.path.exists(output_mesh_model_path):
                    print(f"\n---Skip {input_mesh_model_path} because it already exists")
                    continue
                if input_mesh_model_path.endswith("_transformed.obj"):
                    print(f"\n---Skip {input_mesh_model_path} because this is transformed model")
                    continue

                mtl_data_path = folder + "/" + model_name + ".mtl"

                # メッシュデータの原点処理
                mesh: trimesh.Trimesh = trimesh.load(input_mesh_model_path)
                to_origin_2d, extents_2d = trimesh.bounds.oriented_bounds_2D(mesh.vertices[:, :2])

                # 同次変換の適用と、重心位置への平行移動
                transform = np.eye(4)
                transform[:2, :2] = to_origin_2d[:2, :2]
                transform[:2, 3] = to_origin_2d[:2, 2]
                mesh.apply_transform(transform)
                # 中心をOBBの中心ではなく、重心に 
                mesh.apply_translation(-mesh.centroid)

                # 処理後のメッシュデータを保存
                mesh.export(output_mesh_model_path)
                print(f"\n---Added Mesh {output_mesh_model_path} from {input_mesh_model_path}")

                # mtlファイルをコピーする
                if os.path.exists(mtl_data_path):
                    output_mtl_data_path = folder + "/" + model_name + "_transformed.mtl"
                    shutil.copy2(mtl_data_path, output_mtl_data_path)
                    print(f"\n---Copied Material {output_mtl_data_path} from {mtl_data_path}")
        
        print(f"\n\nTranslation Completed in {folder}...")

        # objデータをusdに変換する
        print(f"\n\nConverting Mesh Data to USD Data in {folder}...")
        (result, models) = omni.client.list(folder)  # convertしたものも含めてモデルを再取得
        for i, entry in enumerate(models):
            model = str(entry.relative_path)
            model_name = os.path.splitext(model)[0]
            model_format = (os.path.splitext(model)[1])[1:]
            # Supported input file formats
            if model_format in supported_file_formats:
                input_model_path = folder + "/" + model
                converted_model_path_base = folder + "/" + "_converted/" + model_name + "_" + model_format
                exts = [".usd", ".usda"]
                for ext in exts:
                    converted_model_path = converted_model_path_base + ext
                    if not os.path.exists(converted_model_path):
                        status = asyncio.get_event_loop().run_until_complete(
                            convert(input_model_path, converted_model_path, load_materials)
                        )
                        if not status:
                            print(f"ERROR Status is {status}")
                        print(f"\n---Added {converted_model_path}")



def clean_up(folders):
    import shutil

    print(f"\n\nCleaning up converted folders...")

    for folder in folders:
        if os.path.exists(folder + "/_converted"):
            shutil.rmtree(folder + "/_converted")
            print(f"---Removed {folder}/_converted")
        else:
            print(f"---Skip removing because no _converted folder in {folder}")

def rename_data_folders(ycb_folder: str):
    """
        もし、データ名にハイフン(-)が含まれている場合、ハイフンをアンダースコア(_)に変換する
    """
    print("\n\n Checking data names and Renaming folders...")
    full_path_of_datas = glob.glob(os.path.join(ycb_folder, "*"))
    data_names = [os.path.split(path)[1] for path in full_path_of_datas]
    data_names.sort()
    for data_name in data_names:
        if "-" in data_name:
            new_data_name = data_name.replace("-", "_")
            os.rename(os.path.join(ycb_folder, data_name), os.path.join(ycb_folder, new_data_name))
            print(f"---Renamed {data_name} to {new_data_name}")
    


if __name__ == "__main__":
    kit = SimulationApp()

    import omni
    from omni.isaac.core.utils.extensions import enable_extension

    enable_extension("omni.kit.asset_converter")
    # コード内で直接指定するフォルダリストとその他の設定
    import glob
    import os


    HOME_PATH = os.environ["HOME"]
    ycb_folder = os.path.join(HOME_PATH, "research_ws/isaac_sim_ws/src/isaac_env/objects/ycb")

    # データ名にハイフンが含まれている場合、アンダースコアに変換
    rename_data_folders(ycb_folder)


    full_path_of_datas = glob.glob(os.path.join(ycb_folder, "*"))
    data_names = [os.path.split(path)[1] for path in full_path_of_datas]
    data_names.sort()
    folders = [os.path.join(ycb_folder, data_name, "google_16k") for data_name in data_names]  

    clean_up(folders)
    asset_convert(folders, max_models=100, load_materials=True)

    # cleanup
    kit.close()
