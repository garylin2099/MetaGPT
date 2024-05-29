#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/6 14:13
@Author  : alexanderwu
@File    : llm_hello_world.py
"""
import asyncio
import json
import re
import os
import nbformat
import shutil

from pathlib import Path
from metagpt.llm import LLM
from metagpt.logs import logger
from metagpt.schema import Message



STRUCTUAL_PROMPT = """
[User Requirement]
{user_requirement}
[code example]
Here is an example of the code: 
{code}

[instruction]
**Summarize**:
[
    "Analyze the data preprocessing and feature selection methods used in the code. Focus on how these methods could potentially improve model performance.",
    "List the specific features processed and discuss their contribution to the model's effectiveness.",
    "Conclude with a brief rationale on the improvement in model performance due to these methods.",
    "Your analysis must be listed in bullet points, with a minimum of 3-5 points(e.g.,1.). Ensure that each point is concise and straightforward.",
]
Output a JSON in the following format:
```json
{{
    "metric": "Report the value of the score.",
    "lower_is_better": "true if the metric should be minimized (e.g., MSE), false if it should be maximized (e.g., accuracy).",
    "Analysis": "Based on the [User Requirement], this concise analysis focuses on the key data processing and feature engineering methods outlined in the code, detailing their potential impact on improving model performance."
}}
"""

REFLECTION_SYSTEM_MSG = """You are an expert in machine learning. Please analyze the provided code and point out the valuable insights to consider."""

async def wrap_code(code: str, lang="python") -> str:
    """Wraps code with three backticks."""
    return f"```{lang}\n{code}\n```"

async def save_rsp_to_file(rsp, filename):
    with open(filename, "w") as file:
            json.dump(rsp, file)

async def load_json_data(json_dir):      
    with open(json_dir, "r") as file:
        json_data = json.load(file)
        return json_data
    
async def clean_json_from_rsp(text):
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        json = "\n".join(matches)  #
        return json
    else:
        return ""
    
# Load the notebook
async def process_notebooks(folder_path,dataset_name : str):
    new_folder_path = Path(folder_path)/f"{dataset_name}_New"
    os.makedirs(new_folder_path, exist_ok=True)
    file_names = os.listdir(folder_path)
    file_names.sort()
    count = 0
    for file_name in file_names:
        original_file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(original_file_path) and file_name != '.DS_Store':
            count+=1
            new_file_name = "{:01d}.ipynb".format((count))
            new_file_path = os.path.join(new_folder_path, new_file_name)
            shutil.copy(original_file_path, new_file_path)
    return new_folder_path

async def load_notebook(new_file_path):           
    with open(new_file_path, 'r', encoding='utf-8') as nb_file:
        notebook = nbformat.read(nb_file, as_version=4)
        all_code_cells = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                all_code_cells.append(cell['source'])
        return all_code_cells


async def format_output(response_dir, response_path):
    new_data = []
    rsp_data = await load_json_data(response_path)
    for i, item in enumerate(rsp_data["analysis"]):
        item_dict = json.loads(item)
        data = {"Analysis": item_dict, "id": i}
        new_data.append(data)

    new_file_path = Path(response_dir) / "response_data_format.json"
    await save_rsp_to_file(new_data, new_file_path)
    return new_file_path

async def analysis_code(data,requirement:str):
    llm = LLM()
    code = data
    structual_prompt = STRUCTUAL_PROMPT.format(
            user_requirement=requirement,
            code = await wrap_code(code) 
    )
    context = llm.format_msg([Message(content=structual_prompt, role="assistant")]) 
            
    llm_response = await llm.aask(
            context, system_msgs=[REFLECTION_SYSTEM_MSG]
        )
    logger.info(llm_response)
    rsp = await clean_json_from_rsp(llm_response)
    return rsp

async def main(requirement:str):
    dataset_name = 'Titanic'
    path  = f"/Users/aurora/Desktop/test_1/{dataset_name}"
    notebook_folder_path = f"/Users/aurora/Desktop/DI experiments/Kaggle_top/{dataset_name}"  # replace with your Kaggle Notebook path
    rsp_directory =  Path(path)/f"Analysis/First_analysis_kaggle/"
    os.makedirs(rsp_directory, exist_ok=True) 
    response_file = Path(rsp_directory)/f"response_data_kaggle_{dataset_name}.json"
    analysis_list = []
    num_files = len([name for name in os.listdir(notebook_folder_path) if name.endswith('.ipynb')])
    new_folder_path = await process_notebooks(notebook_folder_path,dataset_name)
    for i in range(1,num_files+1):
        notebook_file_path = Path(new_folder_path)/f"{i}.ipynb"
        code = await load_notebook(notebook_file_path)
        analysis_rsp = await analysis_code(code,requirement)
        analysis_list.append(analysis_rsp)
    response_dict = {"analysis": analysis_list}
    await save_rsp_to_file(response_dict,response_file)
    final_path = await format_output(rsp_directory, response_file)
    print("The results of the analysis have been saved at:", final_path)
    return final_path

if __name__ == "__main__":
    user_requirement = f"This is a titanic passenger survival dataset, your goal is to predict passenger survival outcome. The target column is Survived.Make sure to generate at least 5 tasks each time, including eda, data preprocessing, feature engineering, model training to predict the target, and model evaluation. Please evaluate their performance separately, and finally input the optimal result. To predict the target, the  Report accuracy on the eval data. Don't need to plot."
    asyncio.run(main(user_requirement))
    
