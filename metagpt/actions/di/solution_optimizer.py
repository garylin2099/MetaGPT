#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/15 19:48
# @Author  : lidanyang
# @File    : solution_optimizer.py
# @Desc    :
import json
import re

from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.schema import Plan, Evaluations, Task
from metagpt.strategy.task_type import TaskType
from metagpt.utils.common import remove_comments, CodeParser

# SOLUTION_OPTIMIZER_PROMPT = """You are a Kaggle competitor who is optimizing the evaluation metric of a given Research Problem through incremental experiments. Now please based on the [Experiments History], propose a new plan that includes only one specific improvement. You have the following information:
# # Research Problem:
# {research_problem}
# # Experiments History:
# {experiments_history}
# {data_info}
# # Available Task Types:
# {task_type_desc}
# # Knowledge Pool (Only for reference)
# Here are some insights proven to be effective in similar but not the same research problems:
# {knowledge_pool}
#
# Follow these instructions and do not forget them:
# - Suggest only one specific improvement per iteration to allow clear evaluation of each change's impact.
# - The exp_iteration of 0 means a baseline model. The exp_iteration of 1 means the first improvement, and so on.
# - The Experiments History includes all plan history and code of each task in the experiment, recorded as cells in a Jupyter notebook. The new plan will be used by programmer to code and run in the same Jupyter notebook.
# - The improvement should be detailed enough to be implemented by the programmer.
# - Initially focus on enhancing feature engineering. If these possibilities are exhausted or exceed 5 attempts (flag by exp_iteration), shift to other types of improvements.
# - Carefully consider which data variables to use, especially if the base feature you want to use is no longer available in the latest dataset or eliminate the impact of ineffective or unsuccessful experiments in history.
# - Don't create any plots.
# - Include model evaluate within the model train task, no need to separate evaluation task.
# - If an output metric is specified in the Research Problem, use it as the optimization target and report it with 6 decimal places.
# - Ensure that the proposed improvements do not introduce data leakage and minimize the risk of overfitting.
#
# # Output:
# Please carefully review all Experiments History, and then response a new plan in the following format:
# [Reflection]: Summarize the previous experiments and their results, analyze what is effective and what is not. Consider how to mitigate the effects of ineffective or unsuccessful experiments in history.
# [Reasoning]: Explain how the proposed improvement will benefit the model.
# [Thought]: Outline the plan for the next experiment iteration including the proposed improvement and any necessary prerequisites.
# Output the new plan in the following list of jsons format, and only start from the change point:
# ```json
# [
#     {{
#         "task_id": str = "unique identifier for a task in plan, can be an ordinal",
#         "dependent_task_ids": list[str] = "ids of tasks prerequisite to this task",
#         "instruction": "what you should do in this task",
#         "task_type": "type of this task, should be one of Available Task Types",
#     }},
#     ...
# ]
# ```
# """
SOLUTION_OPTIMIZER_PROMPT = """You are a Kaggle competitor who is optimizing {metric} score of a given Research Problem through incremental experiments. Now please based on following information, append new tasks that includes only one specific improvement.
# Research Problem:
{research_problem}
# Best Experiments History:
{best_experiment}
# Ineffective Thoughts:
{ineffective_thoughts}
{data_info}
# Available Task Types:
{task_type_desc}
# Knowledge Pool (Only for reference)
Here are some insights proven to be effective in similar but not the same research problems:
{knowledge_pool}

Follow these instructions and do not forget them:
- Suggest only one specific improvement per iteration based on Best Experiment to allow clear evaluation of each change's impact.
- The exp_iteration of 0 means a baseline model. The exp_iteration of 1 means the first improvement, and so on. And now you are in {iteration} iteration.
- The Best Experiment records the most optimal solution up to now, where the code is recorded as cells in a Jupyter notebook. The new task will be implemented and run by the programmer in the same Jupyter notebook.
- The improvement should be detailed enough to be implemented by the programmer.
- The Ineffective Thoughts encompasses all attempts at improvement you made that have been confirmed to not enhance the model's performance.
- Initially focus on enhancing feature engineering. If exceed 5 attempts (flag by exp_iteration, include ineffective attempts), shift to other types of improvements.
- Don't create any plots.
- Ensure that the proposed improvements do not introduce data leakage and minimize the risk of overfitting.
- Avoid using the label column to create features, except for cat encoding.
- Avoid make same feature that already make in previous tasks.
- Include model evaluate within the model train task, and report metric score with 6 decimal places.
- Starting from task_id = "{new_task_id}" strictly in your response.

# Output:
Please carefully review all Experiments History, and then response a new plan in the following format:
[Reflection]: Summarize the previous experiments and their results, analyze what is effective and what is not.
[Reasoning]: Explain how the proposed improvement will benefit the model.
[Thought]: Outline the tasks for the next experiment iteration including the proposed improvement and any necessary prerequisites.
Output new tasks in the following list of jsons format starting from task_id = "{new_task_id}":
```json
[
    {{
        "task_id": str = "unique identifier for a task in plan, can be an ordinal",
        "dependent_task_ids": list[str] = "ids of tasks prerequisite to this task",
        "instruction": "what you should do in this task",
        "task_type": "type of this task, should be one of Available Task Types",
    }},
    ...
]
```
"""


class SolutionOptimizer(Action):
    async def run(
        self,
        plan: Plan,
        knowledge_pool: str,
        data_info: str,
        iteration: int,
        evaluations: Evaluations,
    ):
        best_tasks = evaluations.find_best_tasks(plan.tasks)
        best_tasks = [format_task_results(task) for task in best_tasks]
        best_experiment = [
            task.model_dump(exclude={"dependent_task_ids", "is_success", "is_finished", "task_type"})
            for task in best_tasks
        ]
        ineffective_thoughts = evaluations.ineffective_thoughts()
        ineffective_thoughts = "\n".join(f"- {thought}" for thought in ineffective_thoughts)
        task_type_desc = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in TaskType])
        prompt = SOLUTION_OPTIMIZER_PROMPT.format(
            metric=evaluations.evaluations[0].metric,
            research_problem=plan.goal,
            best_experiment=best_experiment,
            ineffective_thoughts=ineffective_thoughts,
            data_info=data_info,
            task_type_desc=task_type_desc,
            knowledge_pool=knowledge_pool,
            iteration=iteration,
            new_task_id=str(len(plan.tasks) + 1),
        )
        logger.info(prompt)
        rsp = await self._aask(prompt)
        logger.info(f"Solution Optimizer response: \n{rsp}")
        thought = re.findall(r"\[Thought\]:\s*(.*)", rsp)[0]
        new_tasks = json.loads(CodeParser.parse_code(block="", text=rsp))
        return thought, new_tasks


def format_task_results(task: Task):
    task.code = remove_comments(task.code)
    if task.task_type in [TaskType.EDA.type_name]:
        task.result = "The result is too long to display."
    else:
        task.result = "\n".join(task.result.split("\n")[-15:])
    return task
