#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/6 14:13
@Author  : alexanderwu
@File    : llm_hello_world.py
"""
import asyncio
import json
import os
import re
import shutil
from pathlib import Path

from metagpt.llm import LLM
from metagpt.logs import logger
from metagpt.schema import Message
from examples.test_experiments import plan_to_code

STRUCTUAL_PROMPT = """
[User Requirement]
{user_requirement}
[code example]
Here is an example of the code: 
{code}
[score]
The score of the example code is: {score}
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

SUMMARIZE_PROMPT = """
[analysis]
{analysis}
[instruction]
**Summarize**:
[
    "From the detailed [analysis] of data preprocessing and feature engineering provided, extract and summarize specific, actionable recommendations that could optimize these processes.",
    "Include clear steps on how to implement these improvements and briefly describe their potential impact on model performance.",
    "Your summarzie must be listed in bullet points, with a minimum of 3-5 points(e.g.,step 1.). Ensure that each point is concise and straightforward.",
]
Output a JSON in the following format:
```json
{{
    "insights": "Based on the detailed [analysis] provided above on code data preprocessing and feature engineering, please summarize specific and actionable recommendations."
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

async def summarize_insights(data):
    llm = LLM()
    analysis = data.get("analysis", [])
    structual_prompt = SUMMARIZE_PROMPT.format(
        analysis=analysis,
    )
    context = llm.format_msg([Message(content=structual_prompt, role="assistant")])
    llm_response = await llm.aask(context, system_msgs=[REFLECTION_SYSTEM_MSG])
    logger.info(llm_response)
    rsp = await clean_json_from_rsp(llm_response)
    return rsp

async def analysis_code(data,requirement:str):
    llm = LLM()
    code = data.get("code", [])
    score = data.get("score", [])
    structual_prompt = STRUCTUAL_PROMPT.format(
        user_requirement=requirement,
        code=await wrap_code(code),
        score=score,
    )
    context = llm.format_msg([Message(content=structual_prompt, role="assistant")])

    llm_response = await llm.aask(context, system_msgs=[REFLECTION_SYSTEM_MSG])
    logger.info(llm_response)
    rsp = await clean_json_from_rsp(llm_response)
    return rsp


async def main(
        score:list,
        first_save_path:str,
        requirement:str
        ):
    dataset_name = "Titanic"
    path = f"/Users/aurora/Desktop/test_1/{dataset_name}" # Replace with your path
    # save_dir = first_save_path.parent
    # print(save_dir) 
    new_plan_path = Path(path) / "First_sample/plan_DI"
    save_directory = Path(path) / "First_sample/code_DI_new/"
    await plan_to_code.process_code_from_plan(first_save_path, new_plan_path, score, save_directory)
    # analysis rsp path
    rsp_directory = Path(path) / "Analysis/Second_analysis_DI"
    os.makedirs(rsp_directory, exist_ok=True)
    response_file = Path(rsp_directory) / "response_data.json"
    # code->analysis
    analysis_list = []
    for i in range(0, len(score)):
        analyze_code_dir = Path(save_directory) / f"code_{i}.json"
        analyze_data = await load_json_data(analyze_code_dir)
        analysis_rsp = await analysis_code(analyze_data,requirement)
        analysis_list.append(analysis_rsp)
    response_dict = {"analysis": analysis_list}
    await save_rsp_to_file(response_dict, response_file)
    await plan_to_code.get_code(save_directory,score)
    # format analysis file
    final_path = await format_output(rsp_directory, response_file)
    print("The results of the analysis have been saved at:", final_path)
    return final_path

if __name__ == "__main__":
    score = [0.8380,0.8324,0.8380]   # Replace with the new score list
    file_path = '/Users/aurora/Desktop/metaGPT_new/MetaGPT/data/output_1' # Replace with your folder path (the path where DI is automatically saved)
    user_requirement = f"This is a titanic passenger survival dataset, your goal is to predict passenger survival outcome. The target column is Survived.Make sure to generate at least 5 tasks each time, including eda, data preprocessing, feature engineering, model training to predict the target, and model evaluation. Please evaluate their performance separately, and finally input the optimal result. To predict the target, the  Report accuracy on the eval data. Don't need to plot."
    asyncio.run(main(score,file_path,user_requirement))
