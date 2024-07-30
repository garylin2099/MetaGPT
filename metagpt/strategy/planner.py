from __future__ import annotations

import json
import os
import pickle

from pydantic import BaseModel, Field

from metagpt.actions.di.ask_review import AskReview, ReviewConst
from metagpt.actions.di.knowledge_extraction import KnowledgeReranker
from metagpt.actions.di.solution_optimizer import SolutionOptimizer
from metagpt.actions.di.write_plan import (
    WritePlan,
    precheck_update_plan_from_rsp,
    update_plan_from_rsp, DataPreview,
)
from metagpt.const import PLANNER_PATH
from metagpt.logs import logger
from metagpt.memory import Memory
from metagpt.schema import Message, Plan, Task, TaskResult, Evaluations
from metagpt.strategy.knowledge_manager import knowledge_manager, KnowledgeManager
from metagpt.strategy.task_type import TaskType
from metagpt.utils.common import remove_comments

STRUCTURAL_CONTEXT = """
## User Requirement
{user_requirement}
## Context
{context}
## Current Plan
{tasks}
## Current Task
{current_task}
"""

PLAN_STATUS = """
## Finished Tasks
### code
```python
{code_written}
```

### execution result
{task_results}

## Current Task
{current_task}

## Task Guidance
Write complete code for 'Current Task'. And avoid duplicating code from 'Finished Tasks', such as repeated import of packages, reading data, etc.
Specifically, {guidance}
{knowledge_prompt}
"""

KNOWLEDGE_PROMPT = """
# Knowledge
{knowledge}
Refer to the provided knowledge while coding. It encapsulates best practices and proven strategies derived from similar past tasks.
"""


class Planner(BaseModel):
    role_id: str = ""
    plan: Plan
    working_memory: Memory = Field(
        default_factory=Memory
    )  # memory for working on each task, discarded each time a task is done
    auto_run: bool = False
    plan_with_knowledge: bool = False
    task_with_knowledge: bool = False
    optimize_with_knowledge: bool = False
    with_data_info: bool = False
    knowledge_rerank: bool = False
    knowledge_manager: KnowledgeManager = knowledge_manager
    knowledge: str = ""
    data_info: str = ""
    column_info: str = ""

    def __init__(self, goal: str = "", plan: Plan = None, **kwargs):
        plan = plan or Plan(goal=goal)
        super().__init__(plan=plan, **kwargs)

    @property
    def current_task(self):
        return self.plan.current_task

    @property
    def current_task_id(self):
        return self.plan.current_task_id

    async def update_plan(self, goal: str = "", max_tasks: int = 3, max_retries: int = 3):
        if goal:
            self.plan = Plan(goal=goal)

        plan_confirmed = False
        while not plan_confirmed:
            if self.with_data_info and not self.data_info:
                self.data_info = await DataPreview().run(requirement=self.plan.goal)
            if self.plan_with_knowledge or self.task_with_knowledge or self.optimize_with_knowledge:
                await self.update_knowledge()
            context = self.get_useful_memories()
            rsp = await WritePlan().run(
                context,
                max_tasks=max_tasks,
                knowledge=self.knowledge if self.plan_with_knowledge else None,
                data_info=self.data_info if self.with_data_info else None
            )
            self.working_memory.add(Message(content=rsp, role="assistant", cause_by=WritePlan))

            # precheck plan before asking reviews
            is_plan_valid, error = precheck_update_plan_from_rsp(rsp, self.plan)
            if not is_plan_valid and max_retries > 0:
                error_msg = f"The generated plan is not valid with error: {error}, try regenerating, remember to generate either the whole plan or the single changed task only"
                logger.warning(error_msg)
                self.working_memory.add(Message(content=error_msg, role="assistant", cause_by=WritePlan))
                max_retries -= 1
                continue

            _, plan_confirmed = await self.ask_review(trigger=ReviewConst.TASK_REVIEW_TRIGGER)

        update_plan_from_rsp(rsp=rsp, current_plan=self.plan)

        self.working_memory.clear()

    async def update_knowledge(self):
        if not self.knowledge:
            self.knowledge = self.knowledge_manager.format_knowledge(threshold=4)
            if self.knowledge_rerank:
                knowledge = await KnowledgeReranker().run(
                    requirement=self.plan.goal,
                    data_info=self.data_info,
                    max_insights=12
                )
                self.knowledge_manager.knowledge_pool = knowledge
                self.knowledge = '\n'.join([f"- {k.knowledge}" for k in knowledge])
            logger.info(f"Knowledge Pool: \n{self.knowledge}")

    async def optimize_plan(self, iteration: int, evaluations: Evaluations):
        thought, new_tasks = await SolutionOptimizer().run(
            self.plan,
            self.knowledge,
            self.column_info,
            iteration,
            evaluations
        )
        new_tasks = [Task(**task_config) for task_config in new_tasks]
        new_tasks = [task.model_copy(update={"exp_iteration": iteration}) for task in new_tasks]
        for task in new_tasks:
            self.plan.append_task(task)
        return thought

    async def process_task_result(
        self,
        task_result: TaskResult,
        current_iter: int = 0,
    ):
        # ask for acceptance, users can other refuse and change tasks in the plan
        review, task_result_confirmed = await self.ask_review(task_result)

        if task_result_confirmed:
            # tick off this task and record progress
            await self.confirm_task(self.current_task, task_result, review)

        elif "redo" in review:
            # Ask the Role to redo this task with help of review feedback,
            # useful when the code run is successful but the procedure or result is not what we want
            pass  # simply pass, not confirming the result

        else:
            if current_iter == 0:
                # update plan according to user's feedback and to take on changed tasks
                await self.update_plan()
            else:
                raise NotImplementedError("Not implemented for current iteration > 0")

    async def ask_review(
        self,
        task_result: TaskResult = None,
        auto_run: bool = None,
        trigger: str = ReviewConst.TASK_REVIEW_TRIGGER,
        review_context_len: int = 5,
    ):
        """
        Ask to review the task result, reviewer needs to provide confirmation or request change.
        If human confirms the task result, then we deem the task completed, regardless of whether the code run succeeds;
        if auto mode, then the code run has to succeed for the task to be considered completed.
        """
        auto_run = auto_run or self.auto_run
        if not auto_run:
            context = self.get_useful_memories()
            review, confirmed = await AskReview().run(
                context=context[-review_context_len:], plan=self.plan, trigger=trigger
            )
            if not confirmed:
                self.working_memory.add(Message(content=review, role="user", cause_by=AskReview))
            return review, confirmed
        confirmed = task_result.is_success if task_result else True
        return "", confirmed

    async def confirm_task(self, task: Task, task_result: TaskResult, review: str):
        task.update_task_result(task_result=task_result)
        self.plan.finish_current_task()
        self.working_memory.clear()

        confirmed_and_more = (
            ReviewConst.CONTINUE_WORDS[0] in review.lower() and review.lower() not in ReviewConst.CONTINUE_WORDS[0]
        )  # "confirm, ... (more content, such as changing downstream tasks)"
        if confirmed_and_more:
            self.working_memory.add(Message(content=review, role="user", cause_by=AskReview))
            await self.update_plan()

    def get_useful_memories(self, task_exclude_field=None) -> list[Message]:
        """find useful memories only to reduce context length and improve performance"""
        user_requirement = self.plan.goal
        context = self.plan.context
        tasks = [task.dict(exclude=task_exclude_field) for task in self.plan.tasks]
        tasks = json.dumps(tasks, indent=4, ensure_ascii=False)
        current_task = self.plan.current_task.json() if self.plan.current_task else {}
        context = STRUCTURAL_CONTEXT.format(
            user_requirement=user_requirement, context=context, tasks=tasks, current_task=current_task
        )
        context_msg = [Message(content=context, role="user")]

        return context_msg + self.working_memory.get()

    def get_plan_status(self, current_iter: int = 0, evaluations: Evaluations = None) -> str:
        # prepare components of a plan status
        finished_tasks = self.get_finished_tasks(current_iter, evaluations)
        code_written = [remove_comments(task.code) for task in finished_tasks]
        code_written = "\n\n".join(code_written)
        # task_results = [task.result for task in finished_tasks]
        # task_results = "\n\n".join(task_results)
        # only show the result of the last task
        task_results = finished_tasks[-1].result if finished_tasks else ""
        task_type_name = self.current_task.task_type
        task_type = TaskType.get_type(task_type_name)
        guidance = task_type.guidance if task_type else ""
        knowledge = ""
        if self.task_with_knowledge:
            if task_type == "feature engineering":
                knowledge = self.knowledge_manager.format_by_category('feature_engineering')
            elif task_type == "model train":
                knowledge = self.knowledge_manager.format_by_category('model_development')
            elif task_type == "model ensemble":
                knowledge = self.knowledge_manager.format_by_category('model_ensemble')

        # combine components in a prompt
        prompt = PLAN_STATUS.format(
            code_written=code_written,
            task_results=task_results,
            current_task=self.current_task.instruction,
            guidance=guidance,
            knowledge_prompt=KNOWLEDGE_PROMPT.format(knowledge=knowledge) if knowledge else "",
        )

        return prompt

    def get_finished_tasks(self, current_iter, evaluations: Evaluations):
        if current_iter > 0:
            his_tasks = [task for task in self.plan.tasks if task.exp_iteration < current_iter]
            best_tasks = evaluations.find_best_tasks(his_tasks)
            finished_tasks = best_tasks + self.plan.get_finished_tasks(iteration=current_iter)
        else:
            finished_tasks = self.plan.get_finished_tasks()
        return finished_tasks

    def save_state(self):
        """Save the state of the Planner to a file"""
        with open(f"{PLANNER_PATH}/planner_state_{self.role_id}.pkl", "wb") as f:
            pickle.dump(self, f)
            logger.info(f"Planner state saved to planner_state_{self.role_id}.pkl")

    @classmethod
    def load_state(cls, role_id):
        """Load the state of the Planner from a file"""
        filename = f"{PLANNER_PATH}/planner_state_{role_id}.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                return pickle.load(file)
        return None
