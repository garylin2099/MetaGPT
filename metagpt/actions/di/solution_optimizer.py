#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/15 19:48
# @Author  : lidanyang
# @File    : solution_optimizer.py
# @Desc    :
from metagpt.actions import Action
from metagpt.schema import Plan
from metagpt.strategy.task_type import TaskType
from metagpt.utils.common import remove_comments

SOLUTION_OPTIMIZER_PROMPT = """You are a Kaggle competitor who is optimizing the evaluation metric of a given Research Problem through incremental experiments. Now please based on the [Experiments History], propose a new plan that includes only one specific improvement. You have the following information:
# Research Problem:
{research_problem}
# Experiments History:
{experiments_history}
{data_info}
# Available Task Types:
{task_type_desc}
# Knowledge Pool (Only for reference)
Here are some insights proven to be effective in similar but not the same research problems:
{knowledge_pool}

Follow these instructions and do not forget them:
- Suggest only one specific improvement per iteration to allow clear evaluation of each change's impact.
- The exp_iteration of 0 means a baseline model. The exp_iteration of 1 means the first improvement, and so on.
- The Experiments History includes all plan history and code of each task in the experiment, recorded as cells in a Jupyter notebook. The new plan will be used by programmer to code and run in the same Jupyter notebook.
- The improvement should be detailed enough to be implemented by the programmer.
- Initially focus on enhancing feature engineering. If these possibilities are exhausted or exceed 5 attempts (flag by exp_iteration), shift to other types of improvements.
- Carefully consider which data variables to use, especially if the base feature you want to use is no longer available in the latest dataset or eliminate the impact of ineffective or unsuccessful experiments in history.
- Don't create any plots.
- Include model evaluate within the model train task, no need to separate evaluation task.
- If an output metric is specified in the Research Problem, use it as the optimization target and report it with 6 decimal places.
- Ensure that the proposed improvements do not introduce data leakage and minimize the risk of overfitting.

# Output:
Please carefully review all Experiments History, and then response a new plan in the following format:
[Reflection]: Summarize the previous experiments and their results, analyze what is effective and what is not. Consider how to mitigate the effects of ineffective or unsuccessful experiments in history.
[Reasoning]: Explain how the proposed improvement will benefit the model.
[Thought]: Outline the plan for the next experiment iteration including the proposed improvement and any necessary prerequisites.
Output the new plan in the following list of jsons format, and only start from the change point:
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


# TODO: 尝试限制每次在一个task中完成优化；
# TODO: 每次优化时基于最优exp，而不是所有exp，需要考虑代码状态恢复，可以重新运行一遍最优exp涉及到的cell;
class SolutionOptimizer(Action):
    async def run(self, plan: Plan, knowledge_pool: str, data_info: str) -> str:
        research_problem = plan.goal
        experiment_history = [task.model_dump(
            exclude={"dependent_task_ids", "is_success", "is_finished"}
        ) for task in plan.tasks]
        for task in experiment_history:
            task["code"] = remove_comments(task["code"])
            if task["task_type"] not in [
                TaskType.MODEL_TRAIN.type_name,
                TaskType.MODEL_EVALUATE.type_name,
                TaskType.MODEL_ENSEMBLE.type_name
            ]:
                task["result"] = "The result is too long to display."
            else:
                task["result"] = "\n".join(task["result"].split("\n")[-15:])
        for task in experiment_history:
            del task["task_type"]

        task_type_desc = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in TaskType])
        prompt = SOLUTION_OPTIMIZER_PROMPT.format(
            research_problem=research_problem,
            experiments_history=experiment_history,
            data_info=data_info,
            task_type_desc=task_type_desc,
            knowledge_pool=knowledge_pool
        )
        rsp = await self._aask(prompt)
        return rsp
