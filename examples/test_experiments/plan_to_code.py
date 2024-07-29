import os
import shutil
from pathlib import Path
import json

async def save_rsp_to_file(rsp, filename):
    with open(filename, "w") as file:
        json.dump(rsp, file)

async def load_json_data(json_dir):
    with open(json_dir, "r") as file:
        json_data = json.load(file)
        return json_data
    
# def process_code_from_plan(output_dir:str="", new_plan_path:str="",code_save_directory:str=""):
#     os.makedirs(new_plan_path, exist_ok=True)
#     file_names = os.listdir(output_dir)
#     file_names.sort()
#     code_pairs = []
#     os.makedirs(code_save_directory, exist_ok=True)
#     # Renaming and copying JSON files with specific content
#     for i, file_name in enumerate(file_names):
#         original_file_path = os.path.join(output_dir, file_name)
#         if os.path.isdir(original_file_path) and "plan.json" in os.listdir(original_file_path):
#             new_file_name = f"{i:04d}.json"
#             new_file_path = os.path.join(new_plan_path, new_file_name)
#             shutil.copy(os.path.join(original_file_path, "plan.json"), new_file_path)

#             # Load JSON and extract code snippets
#             json_data = load_json_data(new_file_path)
#             code = [json_data["task_map"][str(j)]["code"] for j in range(1, len(json_data["task_map"]) + 1)]
#             code_pairs.append(code)
#             # Save to new location with scores
#             data = {"code": code}
#             new_filename = os.path.join(code_save_directory, f"code_{i}.json")
#             save_rsp_to_file(data, new_filename)
#             print("The code is saved as:",Path(code_save_directory)/new_filename)
            
# async def process_code_from_plan(original_dir, new_folder_path, score, save_directory):
async def process_code_from_plan(output_dir:str="", new_plan_path:str="",score:list=[],code_save_directory:str=""):
    os.makedirs(new_plan_path, exist_ok=True)
    file_names = os.listdir(output_dir)
    file_names.sort()
    code_pairs = []
    os.makedirs(code_save_directory, exist_ok=True)
    # Renaming and copying JSON files with specific content
    for i, file_name in enumerate(file_names):
        original_file_path = os.path.join(output_dir, file_name)
        if os.path.isdir(original_file_path) and "plan.json" in os.listdir(original_file_path):
            new_file_name = f"{i+1:04d}.json"
            new_file_path = os.path.join(new_plan_path, new_file_name)
            shutil.copy(os.path.join(original_file_path, "plan.json"), new_file_path)

            # Load JSON and extract code snippets
            json_data = await load_json_data(new_file_path)
            code = [json_data["task_map"][str(j)]["code"] for j in range(1, len(json_data["task_map"]) + 1)]
            code_pairs.append(code)
            # Save to new location with scores
            data = {"code": code, "score": score[i]}
            new_filename = os.path.join(code_save_directory, f"code_{i}.json")
            await save_rsp_to_file(data, new_filename)
            print("The code is saved as:",Path(code_save_directory)/new_filename)

async def get_code(save_directory,score):
    directory = Path(save_directory)/"code_DI"
    os.makedirs(directory, exist_ok=True) 
    for i in range (0,len(score)):
        response_path =Path(save_directory)/f"code_{i}.json"
        code_str = "" 
        with open(response_path,'r',encoding='utf-8') as file:
            rsp_data = json.load(file)
            for line in rsp_data['code']:
                code_str += line
            output_file_path = Path(directory)/f"code_{i}.py"
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(code_str)