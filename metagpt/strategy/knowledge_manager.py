#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/18 15:57
# @Author  : lidanyang
# @File    : knowledge_manager.py
# @Desc    :
import json
import os
from pathlib import Path
from typing import List, Union

from pydantic import BaseModel

from metagpt.const import KNOWLEDGE_PATH


class Knowledge(BaseModel):
    category: str
    id: str
    knowledge: str
    score: int

    def unique_key(self):
        return self.category, self.id


class KnowledgeManager(BaseModel):
    knowledge_path: Union[str, Path]
    knowledge_pool: List[Knowledge] = []

    def __init__(self, knowledge_path: str):
        super().__init__(knowledge_path=knowledge_path)
        self.knowledge_pool = self.load()

    def load(self):
        """Load the knowledge pool from a JSON file."""
        os.makedirs(os.path.dirname(self.knowledge_path), exist_ok=True)
        if os.path.exists(self.knowledge_path):
            with open(self.knowledge_path, "r", encoding="utf-8") as f:
                return [Knowledge(**data) for data in json.load(f)]
        return []

    def save(self):
        """Save the knowledge pool to a JSON file."""
        knowledge_data = [k.model_dump() for k in self.knowledge_pool]
        with open(self.knowledge_path, "w", encoding="utf-8") as f:
            json.dump(knowledge_data, f, ensure_ascii=False, indent=4)

    def add_knowledge(self, knowledge: Knowledge):
        """Add a knowledge item to the knowledge pool."""
        self.knowledge_pool.append(knowledge)

    def prune(self, threshold: int):
        """
        Remove knowledge items with scores below the specified threshold.

        Args:
            threshold: The minimum score required for a knowledge item to remain in the pool.
        """
        self.knowledge_pool = [k for k in self.knowledge_pool if k.score >= threshold]

    def reindex(self):
        """Reindex the knowledge pool so each category's IDs are sequential starting from 1."""
        category_index = {}
        for knowledge in self.knowledge_pool:
            if knowledge.category not in category_index:
                category_index[knowledge.category] = 1
            knowledge.id = str(category_index[knowledge.category])
            category_index[knowledge.category] += 1

    def format_knowledge(self, threshold: int = None) -> str:
        """Format the knowledge pool as a string."""
        if threshold:
            self.prune(threshold)
        knowledge_by_category = {}
        for knowledge in self.knowledge_pool:
            if knowledge.category not in knowledge_by_category:
                knowledge_by_category[knowledge.category] = []
            knowledge_by_category[knowledge.category].append(knowledge)

        knowledge_items = []
        categories = ['feature_engineering', 'model_development', 'model_ensemble', 'other']
        for category in categories:
            if category in knowledge_by_category:
                knowledge_items.append(
                    f"**{category}**:\n"
                    + "\n".join(
                        f"- {k.knowledge}" for k in knowledge_by_category[category]
                    )
                )

        return "\n".join(knowledge_items)

    def format_by_category(self, category: str, threshold: int = None) -> str:
        """Format the knowledge pool as a string by category."""
        if threshold:
            self.prune(threshold)
        knowledge_list = self.get_knowledge_by_category(category)
        return "\n".join(f"- {k.knowledge}" for k in knowledge_list)

    def get_knowledge_by_category(self, category: str) -> List[Knowledge]:
        """Retrieve all knowledge entries of a specific category."""
        return [k for k in self.knowledge_pool if k.category == category]


knowledge_manager = KnowledgeManager(KNOWLEDGE_PATH / "knowledge_pool_0506_with_kaggle_3_fix.json")
