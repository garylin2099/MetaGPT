import asyncio
import time
import uuid

import nbformat
from pydantic import BaseModel

from metagpt.actions.di.execute_nb_code import ExecuteNbCode
from metagpt.logs import logger
from metagpt.roles.di.data_interpreter import DataInterpreter

DATASETS = [
    "icr",
    "house_prices",
    "titanic",
    # "santander_value_prediction",
    # "santander_customer_transaction",
]


class ExpConfig(BaseModel):
    use_tools: bool = True
    plan_with_knowledge: bool = True
    task_with_knowledge: bool = True
    optimize_with_knowledge: bool = False
    with_data_info: bool = False
    knowledge_rerank: bool = False
    max_optimize_iter: int = 10


async def main(requirement: str, dataset_name: str, config: ExpConfig):
    use_tools = config.use_tools
    execute_code = ExecuteNbCode(nb=nbformat.v4.new_notebook(), timeout=3600)
    role = DataInterpreter(
        execute_code=execute_code,
        use_reflection=True,
        tools=["<all>"] if use_tools else [],
        plan_with_knowledge=config.plan_with_knowledge,
        task_with_knowledge=config.task_with_knowledge,
        optimize_with_knowledge=config.optimize_with_knowledge,
        dataset_name=dataset_name,
        with_data_info=config.with_data_info,
        knowledge_rerank=config.knowledge_rerank,
        max_optimize_iter=config.max_optimize_iter,
        role_id=str(uuid.uuid4()),
    )
    await role.run(requirement)


def get_requirement(dataset_name: str):
    datasets = {
        "house_prices": {
            "path": "/Users/lidanyang/deepw/code/ml_engineer/dev/data_agents_opt/data/house-prices-advanced-regression-techniques",
            "requirement": "This is a house price dataset, your goal is to predict the sale price of a property based on its features. The target column is SalePrice. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. Report RMSE between the logarithm of the predicted value and the logarithm of the observed sales price on the eval data. Train data path: '{train_path}', eval data path: '{eval_path}'.",
        },
        "icr": {
            "path": "/Users/lidanyang/deepw/code/ml_engineer/dev/data_agents_opt/data/icr-identify-age-related-conditions",
            "requirement": "This is a medical dataset with over fifty anonymized health characteristics linked to three age-related conditions. Your goal is to predict whether a subject has or has not been diagnosed with one of these conditions.The target column is Class. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. Report f1 score on the eval data. Train data path: '{train_path}', eval data path: '{eval_path}'.",
        },
        "titanic": {
            "path": "/Users/lidanyang/deepw/code/ml_engineer/dev/data_agents_opt/data/titanic",
            "requirement": "This is a titanic passenger survival dataset, your goal is to predict passenger survival outcome. The target column is Survived. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. Report accuracy on the eval data. Train data path: '{train_path}', eval data path: '{eval_path}'.",
        },
        "santander_customer_transaction": {
            "path": "/Users/lidanyang/deepw/code/ml_engineer/dev/data_agents_opt/data/santander-customer-transaction-prediction",
            "requirement": "This is a customers financial dataset. Your goal is to predict which customers will make a specific transaction in the future. The target column is target. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. Report AUC on the eval data. Train data path: '{train_path}', eval data path: '{eval_path}'.",
        },
        "santander_value_prediction": {
            "path": "/Users/lidanyang/deepw/code/ml_engineer/dev/data_agents_opt/data/santander-value-prediction-challenge",
            "requirement": "This is a customers financial dataset. Your goal is to predict the value of transactions for each potential customer. The target column is target. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. Report RMSLE on the eval data. Train data path: '{train_path}', eval data path: '{eval_path}'.",
        },
    }

    if dataset_name not in datasets:
        print("Dataset not found!")
        return

    data_path = datasets[dataset_name]["path"]
    train_path = f"{data_path}/split_train.csv"
    eval_path = f"{data_path}/split_eval.csv"
    requirement = datasets[dataset_name]["requirement"].format(
        train_path=train_path, eval_path=eval_path
    )
    return requirement


def run_experiment():
    config_op_1 = ExpConfig(
        use_tools=False,
        plan_with_knowledge=False,
        task_with_knowledge=False,
        optimize_with_knowledge=True,
        with_data_info=True,
        knowledge_rerank=True,
    )
    config_op_2 = ExpConfig(
        use_tools=False,
        plan_with_knowledge=False,
        task_with_knowledge=False,
        optimize_with_knowledge=False,
        with_data_info=True,
        knowledge_rerank=False,
    )
    configs = [
        config_op_1,
        config_op_2,
    ]

    n_iter = 1

    for cfg in configs:
        for i in range(n_iter):
            for dataset in DATASETS:
                logger.info(f"Experiment {i + 1} on {dataset} is running.")
                logger.info(f"Config: {cfg}")
                requirement = get_requirement(dataset)
                logger.info(f"Requirement: {requirement}")
                asyncio.run(main(requirement, dataset, cfg))
                logger.info(f"Experiment {i + 1} on {dataset} is done.")
                time.sleep(30)


if __name__ == "__main__":
    run_experiment()
