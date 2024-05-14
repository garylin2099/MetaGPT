import asyncio
import json
import random

from pathlib import Path
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.logs import logger
from metagpt.utils.recovery_util import save_history


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
    #     print(data['Data Preprocessing']['Insights'])
    _data ={
        'Data Preprocessing':{
            'Insights':data['Data Preprocessing']['Insights']
        },
        'Feature Engineering':{
            'Insights':data['Feature Engineering']['Insights']
        },
        'Model Training':{
                'Insights':data['Model Training']['Insights']
        }
    }
    new_data.append(_data)
    return new_data
    # new_file_path = f"/Users/aurora/Desktop/summarize/Second_summarize_DI_2/summarize_new_{i}.json"
    # with open(new_file_path, "w") as file:
    #     json.dump(new_data, file)   

# async def main(requirement: str):
#     preprocess_tools = ["FillMissingValue","MinMaxScale","StandardScale","MaxAbsScale","RobustScale","OrdinalEncode","OneHotEncode","LabelEncode"]
#     feature_engineering_tools = ["PolynomialExpansion","CatCount","TargetMeanEncoder","KFoldTargetMeanEncoder","CatCross","GroupStat","SplitBins","ExtractTimeComps","GeneralSelection","TreeBasedSelection","VarianceBasedSelection"]
#     tools = []
#     tools.extend(await random_sample_tools(preprocess_tools, 2))
#     tools.extend(await random_sample_tools(feature_engineering_tools, 2))
#     print("The tools are:",tools)
#     role = DataInterpreter(use_reflection=True, tools = tools)
#     rsp = await role.run(requirement)
#     logger.info(rsp)
#     save_history(role=role)
# async def clear_history():
    
async def main(requirement: str):
    role = DataInterpreter(use_reflection=True, tools=["<all>"])
    rsp = await role.run(requirement)
    logger.info(rsp)
    save_history(role=role)
    
if __name__ == "__main__":
    #House Price
    data_path = "/Users/aurora/Desktop/ml_benchmark/05_house-prices-advanced-regression-techniques" 
    train_path = f"{data_path}/split_train.csv"
    eval_path = f"{data_path}/split_eval.csv"
    user_requirement = (
        f"This is a house price dataset, your goal is to predict the sale price of a property based on its features. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. Report RMSE between the logarithm of the predicted value and the logarithm of the observed sale prices on the eval data. The target column is 'SalePrice'. Don't transform skewed target column. Don't need to plot. "
        )
    code_path = "/Users/aurora/Desktop/metaGPT_new/MetaGPT/examples/di/first_best_code_house_price copy.py" #baseline_code path
    path = "/Users/aurora/Desktop"  
    directory = Path(path)/f"Summarize_new_test/Second_summarize_DI_0/"
    i = 4
    file_path = Path(directory)/f"summarize_{i}.json" #summarize_res_file_path in "random_sample_analysis.py"
    analysis = load_analysis(file_path)
    baseline_code = load_baseline_code(code_path)
    query = user_requirement+"\n"+"Here are some insights derived from high-performance code:"+ str(analysis)+ "\n" + "Please generate new complete code referenced on the provided baseline code below:\n"+ baseline_code +"\n"+f"Train data path: {train_path}', eval data path: '{eval_path}'."  
    print(query)
    asyncio.run(main(query))

##Titanic
# if __name__ == "__main__":
#     #data_path = "your/path/to/titanic"
#     data_path = "/Users/aurora/Desktop/ml_benchmark/04_titanic" 
#     train_path = f"{data_path}/split_train.csv"
#     eval_path = f"{data_path}/split_eval.csv"
#     requirement = f"This is a titanic passenger survival dataset, your goal is to predict passenger survival outcome. The target column is Survived. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target.Please recommend at least five models and evaluate their performance separately, and finally input the optimal result.  To predict the target, the  Report accuracy on the eval data. Train data path: '{train_path}', eval data path: '{eval_path}'."
#ICR
# if __name__ == "__main__":
#     #data_path = "your/path/to/titanic"
#     data_path = "/Users/aurora/Desktop/ml_benchmark/07_icr-identify-age-related-conditions" 
#     train_path = f"{data_path}/split_train.csv"
#     eval_path = f"{data_path}/split_eval.csv"
#     requirement = f" ICR dataset is a medical dataset with over fifty anonymized health characteristics linked to three age-related conditions. Your goal is to predict whether a subject has or has not been diagnosed with one of these conditions. The target column is Class. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. Report F1 Score on the eval data. Train data path: '{train_path}', eval data path: '{eval_path}'. "

