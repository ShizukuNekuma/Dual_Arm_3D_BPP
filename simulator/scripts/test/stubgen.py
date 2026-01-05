import os
import subprocess

# def generate_stubs_in_directory(base_path, output_base_path):
#     # for root, dirs, files in os.walk(base_path):
#     #     if '__init__.py' in files:
#     #         # rootは__init__.pyを含むディレクトリ
#     #         relative_path = os.path.relpath(root, base_path)

#     #         path = os.path.join(root)
#     #         print(root)
#     #         print(relative_path)
            
#     #         output_path = os.path.join(output_base_path, os.path.dirname(relative_path))
#     #         os.makedirs(output_path, exist_ok=True)

#     #         try:
#     #             result = subprocess.run(
#     #                 ['stubgen', '-o', output_path, root],
#     #                 check=True,
#     #                 stdout=subprocess.PIPE,
#     #                 stderr=subprocess.PIPE,
#     #                 text=True
#     #             )
#     #             print(f"Stub generation successful for path: {path}")
#     #             print("Output:\n", result.stdout)
#     #             if result.stderr:
#     #                 print("Warnings/Errors:\n", result.stderr)
#     #         except subprocess.CalledProcessError as e:
#     #             print(f"An error occurred while processing {relative_path}: {e.stderr}")
#             output_path = os.path.join(output_base_path, os.path.basename(base_path))
#             os.makedirs(output_path, exist_ok=True)
#             try:
#                 result = subprocess.run(
#                     ['stubgen', '-o', output_path, base_path],
#                     check=True,
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.PIPE,
#                     text=True
#                 )
#                 print(f"Stub generation successful for path: {base_path}")
#                 print("Output:\n", result.stdout)
#                 if result.stderr:
#                     print("Warnings/Errors:\n", result.stderr)
#             except subprocess.CalledProcessError as e:
#                 print(f"An error occurred while processing {base_path}: {e.stderr}")


# # 使用例
# workspace_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# package_path = os.path.join(workspace_path, '.venv/lib/python3.10/site-packages/isaacsim/exts/omni.exporter.urdf')
# output_base_path = 'out'
# # 指定されたパスからスタブ生成を行う
# generate_stubs_in_directory(package_path, output_base_path)



# if __name__ == "__main__":

#     from isaacsim import SimulationApp

#     simulation_app = SimulationApp({"headless": True})

#     from mypy import stubgen
#     import sys
#     import omni.isaac.dynamic_control._dynamic_control as dc


#     aaaaa = ['--output', "out", "-p", "omni.isaac.dynamic_control"]


#     stubgen.main(aaaaa)

# while True:
#     pass

import requests
from bs4 import BeautifulSoup

# Fetch the API documentation page
url = "https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.dynamic_control/docs/index.html"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Function to process class and method information
def parse_class_methods(soup):
    pyi_content = []
    
    classes = soup.find_all('dl', class_='py class')
    for cls in classes:
        class_name = cls.find('dt').get_text(strip=True).replace("", "")
        pyi_content.append(f'class {class_name}:\n')
        
        methods = cls.find_all('dl', class_='py method')
        for method in methods:
            method_name = method.find('dt').get_text(strip=True).replace("", "")
            pyi_content.append(f'    def {method_name}(self) -> None:\n        ...\n')
    
    return pyi_content

# Generate .pyi content
pyi_content = parse_class_methods(soup)

# Write to .pyi file
with open('dynamic_control.pyi', 'w') as file:
    file.write('\n'.join(pyi_content))

print("Generated dynamic_control.pyi file successfully.")
