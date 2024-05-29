import asyncio
import json
import random

from pathlib import Path
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.logs import logger
# from metagpt.utils.recovery_util import save_history
from examples.test_experiments import save_plan,plan_to_code

def wrap_code(code: str, lang="python") -> str:
    """Wraps code with three backticks."""
    return f"```{lang}\n{code}\n```"

async def random_sample_tools(tools, num_samples):
    return random.sample(tools, num_samples)

def load_baseline_code(file_path):
    code_str = ""
    with open(file_path, "r") as file:
        for line in file:
            code_str += line
    return code_str 

def load_json_data(json_dir):      
    with open(json_dir, "r") as file:
        json_data = json.load(file)
        return json_data

def load_analysis(file_path):
    new_data = []
    json_data = load_json_data(file_path)
    data = json_data[0]['Analysis']
    _data ={
        'Data Preprocessing':{
            'Data Preprocessing Insights':data['Data Preprocessing']['Insights']
        },
        'Feature Engineering':{
            'Feature Engineering Insights':data['Feature Engineering']['Insights']
        },
        'Model Training':{
            'Model Training Insights':data['Model Training']['Insights']
        }
    }
    new_data.append(_data)
    return new_data

async def main(
        score:list,
        path:str,
        sum_directory:str,
        baseline_code_path:str,
        data_path:str,
        requirement:str = ""
        ):
    #role = DataInterpreter(use_reflection=True, tools=["<all>"])
    #House Price
    train_path = f"{data_path}/split_train.csv"
    eval_path = f"{data_path}/split_eval.csv"
    for i in range(0,5):
        role = DataInterpreter(use_reflection=True)
        file_path = Path(sum_directory)/f"summarize_{i}.json" 
        analysis = load_analysis(file_path)
        insights = {
            'Data Preprocessing': analysis[0]['Data Preprocessing'],
            'Feature Engineering': analysis[0]['Feature Engineering'],
            'Model Training': analysis[0]['Model Training']
        }
        baseline_code = load_baseline_code(baseline_code_path)
        output_requirement = "Please generate new code based on the user requirement, provided insights, and the best-performing baseline code from the previous round. The new code should include both the new changes and the original unchanged code to ensure completeness. The insights consist of three parts: data preprocessing, feature engineering, and model training. Ensure that only the part mentioned in the insight is modified in the newly generated code, while the rest parts should remain unchanged and generate code exactly the same as the baseline code for their respective tasks."
        for insight_name, insight in insights.items():
            query = (
                f"{requirement}\n{insight}\nHere is baseline code:\n{baseline_code}\n{output_requirement} "
                f"Please make sure you have loaded the data and some basic libraries before starting data preprocessing(In {insight_name.lower()} task): Train data path: {train_path}, eval data path: {eval_path}."
            )
            print(query)
            rsp = await role.run(query)
            logger.info(rsp)
            save_path = save_plan.save_history(role=role,save_dir = path,round ='Second')
            new_plan_path = Path(path)/"Second_sample/DI Second (eval)/plan_DI"
            code_save_directory = Path(path)/"Second_sample/DI Second (eval)/code_DI"
            save_directory = Path(path) / "First_sample/code_DI_new/"
            await plan_to_code.process_code_from_plan(save_path,new_plan_path,code_save_directory,score)
            await plan_to_code.get_code(save_directory,score)
        return save_path
    
if __name__ == "__main__":
    score = [0.8380,0.8324,0.8380] #replace with new score list
    dataset_name = "Titanic"
    path = f"/Users/aurora/Desktop/test_1/{dataset_name}" 
    sum_directory = Path(path)/f"summarize/w baseline/test" #summarize_rsp_file_path in "random_sample_analysis.py"
    baseline_code_path = "/Users/aurora/Desktop/test/Titanic/First_sample/code_DI_new/DI Second (eval)/code_DI/code_0.py" #baseline_code path
    data_path = "/Users/aurora/Desktop/ml_benchmark/04_titanic" 
    user_requirement = f"This is a titanic passenger survival dataset, your goal is to predict passenger survival outcome. The target column is Survived.Make sure to generate at least 5 tasks each time, including eda, data preprocessing, feature engineering, model training to predict the target, and model evaluation. Please evaluate their performance separately, and finally input the optimal result. To predict the target, the  Report accuracy on the eval data.Don't need to plot."
    asyncio.run(main(score,path,sum_directory,baseline_code_path,data_path,user_requirement))
    


