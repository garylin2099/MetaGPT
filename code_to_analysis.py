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


async def process_code_from_plan(original_dir, new_folder_path, score, save_directory):
    os.makedirs(new_folder_path, exist_ok=True)
    file_names = os.listdir(original_dir)
    file_names.sort()
    code_pairs = []
    os.makedirs(save_directory, exist_ok=True)
    # Renaming and copying JSON files with specific content
    for i, file_name in enumerate(file_names):
        original_file_path = os.path.join(original_dir, file_name)
        if os.path.isdir(original_file_path) and "plan.json" in os.listdir(original_file_path):
            new_file_name = f"{i+1:04d}.json"
            new_file_path = os.path.join(new_folder_path, new_file_name)
            shutil.copy(os.path.join(original_file_path, "plan.json"), new_file_path)

            # Load JSON and extract code snippets
            json_data = await load_json_data(new_file_path)
            code = [json_data["task_map"][str(j)]["code"] for j in range(1, len(json_data["task_map"]) + 1)]
            code_pairs.append(code)
            # Save to new location with scores
            data = {"code": code, "score": score[i - 1]}
            new_filename = os.path.join(save_directory, f"code_{i}.json")
            await save_rsp_to_file(data, new_filename)


async def format_output(response_dir, response_path):
    new_data = []
    rsp_data = await load_json_data(response_path)
    for i, item in enumerate(rsp_data["analysis"]):
        item_dict = json.loads(item)
        data = {"Analysis": item_dict, "id": i}
        new_data.append(data)

    new_file_path = Path(response_dir) / "response_data_format_1.json"
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


async def analysis_code(data):
    llm = LLM()
    code = data.get("code", [])
    score = data.get("score", [])
    ##ICR dataset##
    # data_path = "/Users/aurora/Desktop/ml_benchmark/07_icr-identify-age-related-conditions"
    # train_path = f"{data_path}/split_train.csv"
    # eval_path = f"{data_path}/split_eval.csv"
    # user_requirement = (
    # f"ICR dataset is a medical dataset with over fifty anonymized health characteristics linked to three age-related conditions. Your goal is to predict whether a subject has or has not been diagnosed with one of these conditions. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. The target column is Class. DON‘T use the target column as feature. Report F1 Score on the eval data. Train data path: '{train_path}', eval data path: '{eval_path}'."
    # )
    # ##Titanic dataset##
    # data_path = "/Users/aurora/Desktop/ml_benchmark/04_titanic"
    # train_path = f"{data_path}/split_train.csv"
    # eval_path = f"{data_path}/split_eval.csv"
    # user_requirement = (
    # f"This is a titanic passenger survival dataset, your goal is to predict passenger survival outcome. The target column is Survived. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. To predict the target, the  Report accuracy on the eval data. Train data path: '{train_path}', eval data path: '{eval_path}'"
    # )
    ##House prince dataset##
    data_path = "/Users/aurora/Desktop/ml_benchmark/05_house-prices-advanced-regression-techniques"
    train_path = f"{data_path}/split_train.csv"
    eval_path = f"{data_path}/split_eval.csv"
    user_requirement = f"This is a house price dataset, your goal is to predict the sale price of a property based on its features. The target column is SalePrice. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. Please recommend at least five models and evaluate their performance separately, and finally input the optimal result. Report RMSE between the logarithm of the predicted value and the logarithm of the observed sales price on the eval data. Train data path: '{train_path}', eval data path: '{eval_path}."

    structual_prompt = STRUCTUAL_PROMPT.format(
        user_requirement=user_requirement,
        code=await wrap_code(code),
        score=score,
    )
    context = llm.format_msg([Message(content=structual_prompt, role="assistant")])

    llm_response = await llm.aask(context, system_msgs=[REFLECTION_SYSTEM_MSG])
    logger.info(llm_response)
    rsp = await clean_json_from_rsp(llm_response)
    return rsp


async def main():
    score = [
        0.1241,
        0.1302,
        0.1313,
        0.1295,
        0.1292,
        0.1242,
        0.1375,
        0.1786,
        0.1567,
        0.1295,
    ]  # Replace with the new score list
    path = "/Users/aurora/Desktop"  # Replace with your path
    original_dir = (
        Path(path) / "metaGPT/MetaGPT/data/output_1"
    )  # Replace with your folder path (the path where DI is automatically saved)

    rename_new_folder_path = Path(path) / "First_sample/plan_DI"
    save_directory = Path(path) / "First_sample/code_DI_new/"
    await process_code_from_plan(original_dir, rename_new_folder_path, score, save_directory)

    # analysis rsp path
    rsp_directory = Path(path) / "Analysis/Second_analysis_DI/"
    os.makedirs(rsp_directory, exist_ok=True)
    response_file = Path(rsp_directory) / "response_data_1.json"

    # code->analysis
    analysis_list = []
    for i in range(1, len(score) + 1):
        analyze_code_dir = Path(save_directory) / f"code_{i}.json"
        analyze_data = await load_json_data(analyze_code_dir)
        analysis_rsp = await analysis_code(analyze_data)
        analysis_list.append(analysis_rsp)
    response_dict = {"analysis": analysis_list}
    await save_rsp_to_file(response_dict, response_file)
    # format analysis file
    final_path = await format_output(rsp_directory, response_file)
    print("The results of the analysis have been saved at:", final_path)


if __name__ == "__main__":
    asyncio.run(main())
