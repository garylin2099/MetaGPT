#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/16 13:56
# @Author  : lidanyang
# @File    : knowledge_extraction.py
# @Desc    :
import json
import os
import random
from datetime import datetime
from typing import List

from metagpt.prompts.di.knowledge_extraction import (
    KNOWLEDGE_EXTRACTION,
    KNOWLEDGE_RERANK,
)
from metagpt.utils.token_counter import count_string_tokens

from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.strategy.knowledge_manager import (
    Knowledge,
    KnowledgeManager,
    knowledge_manager,
)
from metagpt.utils.common import CodeParser
from metagpt.utils.kaggle_client import KaggleClient
from metagpt.utils.read_document import NotebookReader


class KaggleKnowledgeExtraction(Action):
    def __init__(
        self,
        n_iter: int = 5,
        n_sample: int = 8,
        max_insights: int = 10,
        knowledge_path: str = None,
    ):
        super().__init__()
        self.n_iter = n_iter
        self.n_sample = n_sample
        self.max_insights = max_insights
        self.knowledge_path = knowledge_path

        self.knowledge_manager = (
            KnowledgeManager(knowledge_path) if knowledge_path else knowledge_manager
        )
        self.kaggle_client = KaggleClient()
        self._init_storage_dirs()

    def _init_storage_dirs(self):
        """Initialize storage directories"""
        path = os.path.dirname(self.knowledge_manager.knowledge_path)
        file_name = os.path.basename(self.knowledge_manager.knowledge_path).split(".")[
            0
        ]
        # create a new session folder to store the snapshots of each iteration
        session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.session_path = os.path.join(path, "snapshots", file_name, session_id)
        os.makedirs(self.session_path, exist_ok=True)

    async def run(self) -> List[Knowledge]:
        if self.n_sample <= 0:
            sampled_solutions = [[] for _ in range(self.n_iter)]
        else:
            sampled_solutions = self.solution_sampling()
            sampled_solutions.insert(0, [])
            sampled_solutions.insert(0, [])

        for idx, group in enumerate(sampled_solutions):
            logger.info(f"Extracting knowledge from group {idx + 1}...")
            solutions = [self.read_solution(s[0]) for s in group]
            metrics = [s[1] for s in group]
            solution_str = "\n".join(
                f"- Solution {i + 1}(Metric: {metrics[i]}):\n{solutions[i]}"
                for i in range(len(solutions))
            )
            current_knowledge = [
                k.model_dump() for k in self.knowledge_manager.knowledge_pool
            ]
            prompt = KNOWLEDGE_EXTRACTION.format(
                all_solutions=solution_str,
                current_knowledge=current_knowledge,
                max_insights=self.max_insights,
            )
            print(count_string_tokens(prompt, "gpt-4-turbo-preview"))
            rsp = await self._aask(prompt)
            rsp = CodeParser.parse_code(block="", text=rsp)
            new_knowledge = json.loads(rsp)
            self.update_knowledge_pool(new_knowledge)
            self.save_snapshot(idx, group, new_knowledge)
        self.knowledge_manager.save()
        return self.knowledge_manager.knowledge_pool

    def solution_sampling(self, replace: bool = False) -> List[List[str]]:
        """
        Randomly sample solutions from the solutions pool.

        Args:
            replace: whether to sample with replacement. Default is False.
        """
        kaggle_solutions = self.kaggle_client.list_local_solutions(with_metric=True)
        n_iter = min(self.n_iter, len(kaggle_solutions) // self.n_sample)
        sampled_solutions = []

        # TODO: 结合token数量决定采样数量，避免超出token限制
        for _ in range(n_iter):
            if len(kaggle_solutions) < self.n_sample:
                logger.info("Not enough solutions to sample from.")
                break

            sample = random.sample(kaggle_solutions, self.n_sample)
            sampled_solutions.append(sample)
            if not replace:
                kaggle_solutions = [s for s in kaggle_solutions if s not in sample]
        return sampled_solutions

    @staticmethod
    def read_solution(solution_path: str, code_only: bool = False):
        """Read the code from the solution file."""
        if solution_path.endswith(".ipynb"):
            reader = NotebookReader(solution_path)
            code_cells = (
                reader.extract_code_cells() if code_only else reader.extract_all_cells()
            )
            code = "\n".join(code_cells)
        else:
            with open(solution_path, "r", encoding="utf-8") as f:
                code = f.read()
        return code

    def update_knowledge_pool(
        self, new_knowledge: List[dict], prune: bool = True, threshold: int = 3
    ):
        """
        Update the knowledge pool with the new knowledge extracted from the solutions.

        Args:
            new_knowledge: the response containing the extracted knowledge.
            prune: whether to prune the knowledge pool by removing entries with scores below a threshold.
            threshold: the score threshold below which entries will be removed.
        """
        new_keys = {(k["category"], k["id"]) for k in new_knowledge}
        self.knowledge_manager.knowledge_pool = [
            k
            for k in self.knowledge_manager.knowledge_pool
            if (k.category, k.id) not in new_keys
        ]
        self.knowledge_manager.knowledge_pool.extend(
            [Knowledge(**k) for k in new_knowledge if k.get("action") != "delete"]
        )

        if prune:
            self.knowledge_manager.prune(threshold)
        self.knowledge_manager.reindex()

    def save_snapshot(
        self, iteration: int, solution_paths: List[str], new_knowledge: List[dict]
    ):
        """
        Save the snapshot of the current iteration.

        Args:
            iteration: the current iteration number.
            solution_paths: the paths of the sampled solutions.
            new_knowledge: the response containing the extracted knowledge.
        """
        snapshot = {
            "iteration": iteration,
            "solution_paths": solution_paths,
            "new_knowledge": new_knowledge,
            "knowledge_pool": [
                k.model_dump() for k in self.knowledge_manager.knowledge_pool
            ],
        }
        snapshot_file = f"{self.session_path}/snapshot_{iteration}.json"
        with open(snapshot_file, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=4)
        logger.info(f"Snapshot saved to {snapshot_file}")

    def revert_to_snapshot(self, snapshot_file: str):
        """Revert the knowledge pool to a previous snapshot."""
        try:
            with open(snapshot_file, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
                self.knowledge_manager.knowledge_pool = [
                    Knowledge(**k) for k in snapshot["knowledge_pool"]
                ]
                logger.info(f"Knowledge pool restored from snapshot {snapshot_file}.")
        except FileNotFoundError:
            logger.error(f"Snapshot file {snapshot_file} not found.")


class KnowledgeReranker(Action):
    def __init__(self, knowledge_path: str = None):
        super().__init__()
        self.knowledge_manager = (
            KnowledgeManager(knowledge_path) if knowledge_path else knowledge_manager
        )

    async def run(
        self, requirement: str, data_info: str = None, max_insights: int = 15
    ) -> List[Knowledge]:
        """
        Rerank the knowledge pool based on the user requirement and data information and select the top insights.

        Args:
            requirement: the user requirement for the reranking.
            data_info: additional information about the data.
            max_insights: the maximum number of insights to select.

        Returns:
            The selected insights.
        """
        prompt = KNOWLEDGE_RERANK.format(
            requirement=requirement,
            data_info=data_info,
            insights=[k.model_dump() for k in self.knowledge_manager.knowledge_pool],
            max_insights=max_insights,
        )
        rsp = await self._aask(prompt)
        rsp_dict = {
            (item["category"], item["id"]): item
            for item in json.loads(CodeParser.parse_code(block="", text=rsp))
        }
        selected_insights = [
            k
            for k in self.knowledge_manager.knowledge_pool
            if (k.category, k.id) in rsp_dict
        ]
        selected_insights.sort(
            key=lambda k: list(rsp_dict.keys()).index((k.category, k.id))
        )
        return selected_insights


if __name__ == "__main__":
    from metagpt.const import KNOWLEDGE_PATH

    async def main():
        action = KaggleKnowledgeExtraction(
            n_iter=5,
            n_sample=6,
            knowledge_path=KNOWLEDGE_PATH / "knowledge_pool_0506_with_kaggle_3.json"
        )
        knowledge_pool = await action.run()
        for k in knowledge_pool:
            print(k)

    import asyncio
    asyncio.run(main())
