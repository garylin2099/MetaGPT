# Prompt for taking on "eda" tasks
EDA_PROMPT = """
The current task is about exploratory data analysis, please note the following:
- Distinguish column types with `select_dtypes` for tailored analysis and visualization, such as correlation.
- Remember to `import numpy as np` before using Numpy functions.
"""

# Prompt for taking on "data_preprocess" tasks
DATA_PREPROCESS_PROMPT = """
The current task is about data preprocessing. If the provided insights are about data preprocessing, please ensure to modify and generate the new code according to the insights mentioned. The new code should include both the new changes and the original unchanged code to ensure completeness. Please keep the tasks and codes for feature engineering and model training unchanged, and don't omit any codes in baseline!
"""

# Prompt for taking on "feature_engineering" tasks
FEATURE_ENGINEERING_PROMPT = """
The current task is about feature engineering. If the provided insights are about feature engineering, please ensure to modify and generate the new code according to the insights mentioned. The new code should include both the new changes and the original unchanged code to ensure completeness.  Please keep the tasks and codes for data preprocessing and model training unchanged, and don't omit any codes in baseline!
"""

# Prompt for taking on "model_train" tasks
MODEL_TRAIN_PROMPT = """
The current task is about training a model. If the provided insights are about model training, please ensure to modify and generate the new code according to the insights mentioned. The new code should include both the new changes and the original unchanged code to ensure completeness. Please keep the tasks and codes for data preprocessing and model feature engineering unchanged, and don't omit any codes in baseline!
"""

# Prompt for taking on "model_evaluate" tasks
MODEL_EVALUATE_PROMPT = """
The current task is about evaluating a model, please note the following:
- Ensure that the evaluated data is same processed as the training data. If not, remember use object in 'Done Tasks' to transform the data.
- Use trained model from previous task result directly, do not mock or reload model yourself.
"""

# Prompt for taking on "image2webpage" tasks
IMAGE2WEBPAGE_PROMPT = """
The current task is about converting image into webpage code. please note the following:
- Single-Step Code Generation: Execute the entire code generation process in a single step, encompassing HTML, CSS, and JavaScript. Avoid fragmenting the code generation into multiple separate steps to maintain consistency and simplify the development workflow.
- Save webpages: Be sure to use the save method provided.
"""
