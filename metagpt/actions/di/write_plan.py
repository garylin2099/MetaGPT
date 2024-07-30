# -*- encoding: utf-8 -*-
"""
@Date    :   2023/11/20 11:24:03
@Author  :   orange-crow
@File    :   plan.py
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Tuple

from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.schema import Message, Plan, Task
from metagpt.strategy.task_type import TaskType
from metagpt.tools.libs.data_preprocess import get_data_info
from metagpt.utils.common import CodeParser


class WritePlan(Action):
    PROMPT_TEMPLATE: str = """
    # Context:
    {context}
    # Available Task Types:
    {task_type_desc}
    # Task:
    Based on the context, write a plan or modify an existing plan of what you should do to achieve the goal. A plan consists of one to {max_tasks} tasks.
    If you are modifying an existing plan, carefully follow the instruction, don't make unnecessary changes. Give the whole plan unless instructed to modify only one task of the plan.
    If you encounter errors on the current task, revise and output the current single task only.
    {data_info_prompt}
    {knowledge_prompt}
    Output a list of jsons following the format:
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
    DATA_INFO_PROMPT: str = """
    # Data Info:
    {data_info}
    - Address specific data challenges outlined in the Data Info and apply targeted strategies in the plan.
    - Interpret the semantic information in Data Info and ascertain the practical implications of columns to align the plan with meaningful data insights.
    """
    KNOWLEDGE_PROMPT: str = """
    # Knowledge Pool:
    {knowledge}
    ## Instruction:
    - Utilize the strategies from the Knowledge Pool aiming for the highest score while writing the plan.
    - Create multiple tasks of the same type from different perspectives only if one task cannot achieve them together.
    - Ensure the tasks are logically connected and contribute to the overall goal.
    """

    async def run(
            self, context: list[Message], max_tasks: int = 5, knowledge: str = None, data_info: str = None
    ) -> str:
        task_type_desc = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in TaskType])
        prompt = self.PROMPT_TEMPLATE.format(
            context="\n".join([str(ct) for ct in context]),
            max_tasks=max_tasks,
            task_type_desc=task_type_desc,
            knowledge_prompt=self.KNOWLEDGE_PROMPT.format(knowledge=knowledge) if knowledge else "",
            data_info_prompt=self.DATA_INFO_PROMPT.format(data_info=data_info) if data_info else ""
        )
        rsp = await self._aask(prompt)
        rsp = CodeParser.parse_code(block='', text=rsp)
        return rsp


class DataPreview(Action):
    async def run(self, requirement: str) -> str:
        prompt = f"""
        Output the train data path and target column name from the requirement: {requirement}
        Output your response in the following format:
        path: <train_data_path>
        target: <target_column_name>
        """
        rsp = await self._aask(prompt)
        path = re.search(r"path: (.+)", rsp).group(1)
        target = re.search(r"target: (.+)", rsp).group(1)
        data_info = get_data_info(path, target)
        return data_info


def update_plan_from_rsp(rsp: str, current_plan: Plan):
    rsp = json.loads(rsp)
    tasks = [Task(**task_config) for task_config in rsp]

    if len(tasks) == 1 or tasks[0].dependent_task_ids:
        if tasks[0].dependent_task_ids and len(tasks) > 1:
            # tasks[0].dependent_task_ids means the generated tasks are not a complete plan
            # for they depend on tasks in the current plan, in this case, we only support updating one task each time
            logger.warning(
                "Current plan will take only the first generated task if the generated tasks are not a complete plan"
            )
        # handle a single task
        if current_plan.has_task_id(tasks[0].task_id):
            # replace an existing task
            current_plan.replace_task(tasks[0])
        else:
            # append one task
            current_plan.append_task(tasks[0])

    else:
        # add tasks in general
        current_plan.add_tasks(tasks)


def precheck_update_plan_from_rsp(rsp: str, current_plan: Plan) -> Tuple[bool, str]:
    temp_plan = deepcopy(current_plan)
    try:
        update_plan_from_rsp(rsp, temp_plan)
        return True, ""
    except Exception as e:
        return False, e
