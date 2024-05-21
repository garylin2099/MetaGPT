#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/30 16:54
# @Author  : lidanyang
# @File    : knowledge_extraction.py
# @Desc    :
KNOWLEDGE_EXTRACTION = """
As an experienced machine learning expert, your task is to merge your deep knowledge with Top Solutions to distill key insights and update Current Knowledge, ultimately aiming to significantly enhance the performance in future competitions.

## Top Solutions:
{all_solutions}

## Current Knowledge: 
{current_knowledge}

## Insight Categories:
- **feature_engineering**: Cover feature creation, transformation, or selection.
- **model_development**: Cover model selection, tuning (explain which parameters to adjust and methods for specific models).
- **model_ensemble**: Suggest specific models to ensemble, like LightGBM and CatBoost, etc. or describe ensemble techniques.
- **other**: Explore other innovative techniques, such as handling data imbalances, etc.

## Actions:
- **add**: Add a novel, previously uncharted new insight with a new ID.
- **modify**: Revise or enhance an existing insight; ID unchanged.

## Instructions:
- Never add insights related to EDA, visualizations, file management, and version monitor or control.
- Focus only on tabular data modality, excluding NLP, CV, and other modalities.
- The knowledge should provide clear instructions and best practices, optionally including suggestions for useful Python libraries when relevant.
- Emphasize detailed insights into feature creation, highlighting critical considerations and methodologies.
- Ensure each new insight offers a fresh perspective without duplicating existing one, keep content concise.
- When relevant, clarify the application context of insights, such as modality (cls, mcls, reg, time-series), data type (num, category, ...), sector (finance, healthcare, ...), etc.
- Score each entry from 0 to 5 based on universality, effectiveness, and correctness.
- Ensure the total number of actions (add+modify+delete) per update between `0` and `{max_insights}`.
- Carefully modify based on your expertise, because the top solutions can not cover all best practices.

## Output:
Output a list of JSON objects with insights categorized and actioned appropriately, ensuring each insight is uniquely numbered within each category. Example format:
```json
[
    {{"category": "feature_engineering", "id": "1", "knowledge": "Updated knowledge", "score": 4, "action": "modify"}},
    {{"category": "model_development", "id": "1", "knowledge": "New knowledge", "score": 3, "action": "add"}},
    ...
]
```
"""

KNOWLEDGE_RERANK = """
As a knowledge selector, you are tasked with enhancing user outcomes by reranking and selecting the most suitable abd helpful insights from the knowledge pool.

# User Requirement:
{requirement}
# Data Information:
{data_info}
# Current Knowledge Pool:
{insights}

# Selection Criteria:
- Relevance to specific challenges indicated by user requirements and detailed data characteristics.

# Output:
Return a list of JSON objects with the top insights, only include the category and id. Ensure the total number of selections don't exceed {max_insights}. Example format:
```json
[
    {{"category": "feature_engineering", "id": "3"}},
    {{"category": "model_development", "id": "4"}},
    ...
]
```
"""
