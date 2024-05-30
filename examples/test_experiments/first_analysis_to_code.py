import asyncio
import json
import random
import pathlib as Path

from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.logs import logger
# from metagpt.utils.recovery_util import save_history
from examples.test_experiments import save_plan

async def load_json_data(json_dir):      
    with open(json_dir, "r") as file:
        json_data = json.load(file)
        return json_data
    
async def random_sample_tools(tools, num_samples):
    return random.sample(tools, num_samples)

async def load_analysis (file_path,i): 
    rsp = await load_json_data(file_path)
    data = str(rsp[i]['Analysis']['Analysis'])
    return data

async def load_summarize(file_path):
    with open(file_path,'r',encoding = 'utf-8') as file:
        rsp = json.load(file)
    return rsp    

async def main(file_path: str,data_path:str,requirement:str = "",save_dir:str = ""):
    #role = DataInterpreter(use_reflection=True, tools=["<all>"])
    train_path = f"{data_path}/split_train.csv"
    eval_path = f"{data_path}/split_eval.csv"
    analysis_file = await load_json_data(file_path)
    for i in range(0,len(analysis_file)):
        role = DataInterpreter(use_reflection=True)
        analysis = await load_analysis(file_path,i)
        query = requirement+"Here are some insights derived from high-performance code:"+ str(analysis) + f"Train data path: '{train_path}', eval data path: '{eval_path}'."
        print(query)
        rsp = await role.run(query)
        logger.info(rsp)
        save_path = save_plan.save_history(role=role,save_dir = save_dir,round ='First')
        print("The results of code is saved as:",save_path)
    return save_path

if __name__ == "__main__":    
    dataset_name = "Titanic"
    save_dir = f"/Users/aurora/Desktop/test_1/{dataset_name}/" #replace with your path
    file_path  = f"/Users/aurora/Desktop/test_1/{dataset_name}/Analysis/First_analysis_kaggle/response_data_format.json"
    #Titanic
    data_path = "/Users/aurora/Desktop/ml_benchmark/04_titanic" 
    user_requirement = f"This is a titanic passenger survival dataset, your goal is to predict passenger survival outcome. The target column is Survived.Make sure to generate at least 5 tasks each time, including eda, data preprocessing, feature engineering, model training to predict the target, and model evaluation. Please evaluate their performance separately, and finally input the optimal result. To predict the target, the  Report accuracy on the eval data. Don't need to plot."
    asyncio.run(main(file_path,data_path,user_requirement,save_dir))


