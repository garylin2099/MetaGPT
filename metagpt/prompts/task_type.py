# Prompt for taking on "eda" tasks
EDA_PROMPT = """
The current task is about exploratory data analysis, please note the following:
- Distinguish column types with `select_dtypes` of `object` or `np.number` for tailored analysis and visualization, such as correlation.
- Remember to `import numpy as np` before using Numpy functions.
- Don't create any plots.
"""

# Prompt for taking on "data_preprocess" tasks
DATA_PREPROCESS_PROMPT = """
The current task is data preprocessing before creating new features, please note the following:
- Monitor data types per column, applying appropriate methods.
- Ensure operations are on existing dataset columns.
- Avoid writing processed data to files.
- Avoid any change to label column.
- Never do data scaling and encoding in this task.
- Each step do data preprocessing to train, must do same for test separately at the same time.
- Always copy the DataFrame before processing it and use the copy to process.
"""

# Prompt for taking on "feature_engineering" tasks
FEATURE_ENGINEERING_PROMPT = """
The current task is about feature engineering. when performing it, please adhere to the following principles:
- Generate as diverse features as possible to improve the model's performance step-by-step. 
- Use available feature engineering tools if they are potential impactful. The result of tool is no need to join back to the original DataFrame because the tool will do it.
- Exclude ID columns from feature generation and remove them.
- Each feature engineering operation performed on the train set must also applies to the test separately at the same time.
- Prefer alternatives to one-hot encoding for categorical data.
- Avoid using the label column to create features, except for cat encoding.
- Use the data from previous task result if exist, do not mock or reload data yourself.
- Always copy the DataFrame before processing it and use the copy to process.
"""

# Prompt for taking on "model_preprocess" tasks
MODEL_PREPROCESS_PROMPT = """
The current task is about preprocessing the data before model training, please note the following:
- Avoid any change to label column.
- Ensure operations are on existing dataset columns.
- Prefer alternatives to one-hot encoding for categorical data.
- Each step do data preprocessing to train, must do same for test separately at the same time.
- Use the data from previous task result directly, do not mock or reload data yourself.
- Always copy the DataFrame before processing it and use the copy to process.
"""

# Prompt for taking on "model_train" tasks
MODEL_TRAIN_PROMPT = """
The current task is about training and evaluate a model, please ensure high performance:
- Keep in mind that your user prioritizes results and is highly focused on model performance. So, when needed, feel free to use models of any complexity to improve effectiveness, such as XGBoost, CatBoost, etc.
- If use model requires no missing like Linear Regression, etc., fill missing values together with all steps.
- Use the data from previous task result directly, do not mock or reload data yourself.
- Set suitable hyperparameters for the model, make metrics as high as possible.
- Ensure that the evaluated data is same processed as the training data. If not, remember use object in 'Done Tasks' to transform the data.
- Update the best model and specific metric so far, not just output the metric of current task.
"""

# Prompt for taking on "model_evaluate" tasks
MODEL_EVALUATE_PROMPT = """
The current task is about evaluating a model, please note the following:
- Ensure that the evaluated data is same processed as the training data. If not, remember use object in 'Done Tasks' to transform the data.
- Use trained model from previous task result directly, do not mock or reload model yourself.
- Update the best model and specific metric so far, not just output the metric of current task.
"""

# Prompt for taking on "model_ensemble" tasks
MODEL_ENSEMBLE_PROMPT = """
The current task is focused on enhancing performance through ensemble models. Please follow these guidelines:
- Start and select top and metric-similar performing models from previous tasks if exist.
- You can also add new models that provide new algorithms or different feature sets to enrich the ensemble.
- Compare the performance of various ensemble strategies if exist and individual models to select the most effective one.
"""

# Prompt for taking on "image2webpage" tasks
IMAGE2WEBPAGE_PROMPT = """
The current task is about converting image into webpage code. please note the following:
- Single-Step Code Generation: Execute the entire code generation process in a single step, encompassing HTML, CSS, and JavaScript. Avoid fragmenting the code generation into multiple separate steps to maintain consistency and simplify the development workflow.
- Save webpages: Be sure to use the save method provided.
"""
