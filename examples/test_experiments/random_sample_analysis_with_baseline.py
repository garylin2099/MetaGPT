import random
import asyncio
import json
import re
import os

from pathlib import Path
from metagpt.llm import LLM
from metagpt.logs import logger
from metagpt.schema import Message

STRUCTUAL_PROMPT = """
[Original Experiences]
Below are 20 randomly sampled experiences from a previous dataset, each labeled with an experience, a corresponding score, and a unique ID:
{experience}
[baseline Code]
Here is the baseline code: {code}
**Task**:
For each stage (Data Preprocessing, Feature Engineering, Model Training):
- Reasoning: Analyze the provided experiences based on the variation in their scores to identify which experiences are most likely to improve baseline code's performance. 
- Reference: Connect each key point with the corresponding experience ID.
- Insights: Based on the given reasons, referenced experiences and the baseline code, provide some as specific and actionable as possible insights you believe can enhance the baseline code's performance. It is best if these insights are based on the provided past experiences and are different from the baseline code. Your insights must be listed in bullet points, with a minimum of 2 points(e.g.,1.). 

**Instructions for Output**:
Organize the output into three sections corresponding to each stage of data handling:
   - Data Preprocessing
   - Feature Engineering
   - Model Training

**Expected Output Format**:
```json
{{
    "Data Preprocessing": {{
        "Source": ["List all experience IDs related to data preprocessing."]
        "Reference IDs": ["List of IDs that you main reference or choose from this stage source."],
        "Reasoning(yes)": "Reasons for selecting these experiences.",
        "Not Reference IDs": ["List of IDs that you did not reference or choose from this stage source."],
        "Reasoning(No)": "Reasons for not selecting these experiences.",
        "Insights": "Based on the reasons and experiences, propose as specific and actionable as possible insights for improving baseline code's performance."
    }},
    "Feature Engineering": {{
        "Source": ["List all experience IDs related to feature engineering."]
        "Reference IDs": ["List of IDs that you main reference or choose from this stage source."],
        "Reasoning(yes)": "Reasons for selecting these experiences.",
        "Not Reference IDs": ["List of IDs that you did not reference or choose from this stage source."],
        "Reasoning(No)": "Reasons for not selecting these experiences.",
        "Insights": "Based on the reasons and experiences, propose as specific and actionable as possible insights for improving baseline code's performance. If you need to create new features, please provide specific feature names."
    }},
    "Model Training": {{
       "Source": ["List all experience IDs related to model training."]
        "Reference IDs": ["List of IDs that you main reference or choose from this stage source."],
        "Reasoning(yes)": "Reasons for selecting these experiences.",
        "Not Reference IDs": ["List of IDs that you did not reference or choose from this stage source."],
        "Reasoning(No)": "Reasons for not selecting these experiences.",
        "Insights": "Based on the reasons and experiences, propose as specific and actionable as possible insights for improving baseline code's performance."
    }}
}}

"""

REFLECTION_SYSTEM_MSG = "As a Kaggle grandmaster participating in a competition, you need to analyze your experiences and propose evolutionary points that are more likely to improve the performance of baseline code."
#, with a minimum of 3 points(e.g.,1.)

async def _random_sample(analysis, num_samples):
    return random.sample(analysis, num_samples)

async def load_json_data(json_dir):      
    with open(json_dir, "r") as file:
        json_data = json.load(file)
        return json_data
    
async def save_rsp_to_file(rsp, filename):
    with open(filename, "w") as file:
            json.dump(rsp, file)
    

async def clean_json_from_rsp(text):
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        json = "\n".join(matches)  #
        return json
    else:
        return ""  
    
async def format_output(rsp):
    rsp_list = []
    new_data = [] 
    rsp_list.append(rsp)              
    for item in rsp_list:
        item_dict = json.loads(item)
        data = {
                "Analysis": item_dict,
        }
        new_data.append(data)
    return new_data

async def get_id_list(random_sample,id_list_file):
    id_list = []
    for m in range(0,len(random_sample)):
        random_sample_id_list = []
        id_list.append(random_sample[m]['id'])
    random_sample_id = {
        "Random_ID":id_list
    }
    random_sample_id_list.append(random_sample_id)
    print("List of ID:",random_sample_id_list)
    await save_rsp_to_file(random_sample_id,id_list_file)

async def get_analysis_pool(json_file):  
    data_list = []
    count = 0
    json_data = await load_json_data(json_file)
    for i in range(0,len(json_data)):
        analysis_data = (json_data[i]['Analysis']['Analysis'])
        for j in range(0,len(json_data[i]['Analysis']['Analysis'])):
            data = {
                "Analysis":analysis_data[j],
                "Score":json_data[i]['Analysis']['metric'],
                "low_is_better":json_data[i]['Analysis']['lower_is_better'],
                "id":count
            }
            count+=1
            data_list.append(data)
    rsp_file = Path(os.path.dirname(json_file))/"analysis_pool_sample.json"
    await save_rsp_to_file(data_list,rsp_file)
    return data_list

async def load_baseline_code(file_path):
    code_str = ""
    with open(file_path, "r") as file:
        for line in file:
            code_str += line
    return code_str    
     
async def summarize_insights(experience,code):
    llm = LLM()
    experiences = experience
    code = code
    structual_prompt = STRUCTUAL_PROMPT.format(
            experience = experiences, 
            code = code,  
    )
    context = llm.format_msg([Message(content=structual_prompt, role="assistant")]) 
    llm_response = await llm.aask(
            context, system_msgs=[REFLECTION_SYSTEM_MSG]
        )
    logger.info(llm_response)
    rsp = await clean_json_from_rsp(llm_response)
    format_rsp = await format_output(rsp)
    return format_rsp

async def main(final_path:str,baseline_code_path:str):
    dataset_name = "Titanic"
    path = f"/Users/aurora/Desktop/test_1/{dataset_name}"
    directory = Path(path)/f"summarize/w baseline/test"
    os.makedirs(directory, exist_ok=True) 
    experience = await get_analysis_pool(final_path)
    baseline_code = await load_baseline_code(baseline_code_path)
    for i in range(0,5):
        summarize_res_file = Path(directory)/f"summarize_{i}.json"
        random_sample = await _random_sample(experience,int(len(experience)/2)) 
        summarize_insights_rsp = await summarize_insights(random_sample,baseline_code)
        await save_rsp_to_file(summarize_insights_rsp,summarize_res_file)
        print("The results of the summarizes have been saved at:",summarize_res_file)
    return directory

if __name__ == "__main__":
    dataset_name = "Titanic"
    path = f"/Users/aurora/Desktop/test_1/{dataset_name}"   #Replace with your path
    final_path = Path(path)/"Analysis/Second_analysis_DI/response_data_format.json" #final_path" in "code_to_analysis.py"
    baseline_code_path = "/Users/aurora/Desktop/test/Titanic/First_sample/code_DI_new/DI Second (eval)/code_DI/code_0.py" #replace with new path
    asyncio.run(main(final_path,baseline_code_path))

    