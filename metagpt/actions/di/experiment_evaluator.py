#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/22 11:03
# @Author  : lidanyang
# @File    : experiment_evaluator.py
# @Desc    :
import json

from metagpt.actions import Action
from metagpt.actions.di.solution_optimizer import format_task_results
from metagpt.logs import logger
from metagpt.schema import Plan
from metagpt.utils.common import CodeParser

EXPERIMENT_EVALUATOR_PROMPT = """You are a evaluator who is responsible for evaluating the best performance of the experiment based on the Research Problem and Task Results.
# Research Problem:
{research_problem}
# Task Results:
{task_results}
Output your response in the following JSON format without any comments:
```json
{{
    "metric": "the metric specified in the Research Problem",
    "is_lower_better": bool = "true if the metric should be minimized, false if it should be maximized",
    "score": float = "the best evaluation metric score you get from the task results",
    "task_id": str = "the task_id which you get the score",
}}
```
"""


class ExperimentEvaluator(Action):
    async def run(self, plan: Plan, current_iter: int) -> dict:
        research_problem = plan.goal
        tasks = [format_task_results(task) for task in plan.tasks]
        task_results = [
            task.model_dump(include={"task_id", "result"})
            for task in tasks
            if task.exp_iteration == current_iter
        ]
        prompt = EXPERIMENT_EVALUATOR_PROMPT.format(
            research_problem=research_problem, task_results=task_results
        )
        rsp = await self._aask(prompt)
        rsp = json.loads(CodeParser.parse_code(block="", text=rsp))
        rsp['exp_iteration'] = current_iter
        logger.info(f"Experiment Evaluator response for iteration {current_iter}: {rsp}")
        return rsp
