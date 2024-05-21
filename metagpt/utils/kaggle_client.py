#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/12 10:49
# @Author  : lidanyang
# @File    : kaggle_client.py
# @Desc    :
import os
from typing import Literal

from pydantic import BaseModel

from metagpt.config2 import config
from metagpt.const import KAGGLE_PATH
from metagpt.logs import logger

os.environ["KAGGLE_USERNAME"] = config.kaggle.username
os.environ["KAGGLE_KEY"] = config.kaggle.key
from kaggle import api


class Competition(BaseModel):
    description: str
    metric: str
    category: str
    tags: list
    url: str
    team_count: int


class KaggleClient:
    def __init__(self, save_path: str = KAGGLE_PATH):
        self.save_path = save_path
        self.competition_path = os.path.join(self.save_path, "competitions")
        self.dataset_path = os.path.join(self.save_path, "datasets")

        self.api = api
        self._init_storage_dirs()

    def _init_storage_dirs(self):
        """Initialize storage directories"""
        os.makedirs(self.competition_path, exist_ok=True)
        os.makedirs(self.dataset_path, exist_ok=True)

    def list_local_solutions(self, with_metric: bool = False):
        """List all Kaggle solution files(notebooks and Python scripts) from local storage."""
        solutions = []
        for root, dirs, files in os.walk(self.competition_path):
            for file in files:
                if file.endswith(".ipynb") or file.endswith(".py"):
                    # 文件的上一级目录名即为比赛名
                    comp_name = os.path.basename(os.path.dirname(root))
                    metric = COMPETITIONS.get(comp_name, None)
                    solutions.append(
                        (os.path.join(root, file), metric)
                        if with_metric
                        else os.path.join(root, file)
                    )
        return solutions

    def get_all_competitions(self, search: str = None, **kwargs):
        # TODO: get all competitions including past competitions
        all_competitions = []
        total_pages = float("inf")
        page = 1

        while page <= total_pages:
            competitions = self.api.competitions_list(
                page=page, search=search, **kwargs
            )
            if not competitions:
                break
            all_competitions.extend(competitions)
            page += 1
        return all_competitions

    def competitions_list(self, page: int = 1, search: str = None, **kwargs):
        """
        List Kaggle competitions

        Args:
            page (int): The page number to fetch, default is 1
            search (str): Search query to filter competitions, default is None
            **kwargs: Additional arguments to pass to Kaggle API
        """
        competitions = self.api.competitions_list(page=page, search=search, **kwargs)
        return [
            Competition(
                description=comp.description,
                metric=comp.evaluationMetric,
                category=comp.category,
                tags=comp.tags,
                url=comp.url,
                team_count=comp.teamCount,
            )
            for comp in competitions
        ]

    def download_competition_data(self, competition: str):
        """
        Download competition data from Kaggle

        Args:
            competition (str): Competition name
        """
        path = os.path.join(self.competition_path, competition, "data")
        os.makedirs(path, exist_ok=True)

        files = self.api.competition_list_files(competition)
        for file in files:
            self.api.competition_download_file(competition, file.name, path)
            logger.info(f"Downloaded competition data {file.name} to {path}")

    def download_dataset_data(self, dataset: str):
        """
        Download dataset data from Kaggle

        Args:
            dataset (str): Dataset name
        """
        path = os.path.join(self.dataset_path, dataset, "data")
        os.makedirs(path, exist_ok=True)

        file_result = self.api.dataset_list_files(dataset)
        files = file_result.files
        for file in files:
            self.api.dataset_download_file(dataset, file.name, path)
            logger.info(f"Downloaded dataset data {file.name} to {path}")

    def download_top_solution(
        self,
        competition: str = None,
        dataset: str = None,
        top_k: int = 5,
        sort_by: Literal[
            "hotness",
            "commentCount",
            "dateCreated",
            "dateRun",
            "relevance",
            "scoreAscending",
            "scoreDescending",
            "viewCount",
            "voteCount",
        ] = "hotness",
        unique_author: bool = True,
        **kwargs,
    ):
        """
        Download top-k code solutions from Kaggle competition or dataset

        Args:
            competition (str): Competition name
            dataset (str): Dataset name, if both competition and dataset are provided, competition will be used
            top_k (int): Number of top notebooks to download
            sort_by (str): Sort by parameter
            unique_author (bool): Whether to download unique author's solutions
            **kwargs: Additional arguments to pass to Kaggle API
        """
        if not competition and not dataset:
            raise ValueError("Either competition or dataset must be provided")

        if competition:
            path = os.path.join(
                self.save_path, "competitions", competition, "notebooks"
            )
        else:
            path = os.path.join(self.save_path, "datasets", dataset, "notebooks")

        os.makedirs(path, exist_ok=True)

        logger.info(
            f"Downloading top {top_k} solutions from {competition or dataset}..."
        )
        kernels = self.api.kernels_list(
            competition=competition,
            dataset=dataset,
            page_size=100,
            sort_by=sort_by,
            language="python",
            **kwargs,
        )
        kernels = sorted(kernels, key=lambda x: x.totalVotes, reverse=True)

        if unique_author:
            best_kernels_per_author = {}
            for kernel in kernels:
                print(kernel.__dict__)
                if len(best_kernels_per_author) >= top_k:
                    break
                author = kernel.author
                if author not in best_kernels_per_author:
                    best_kernels_per_author[author] = kernel
            top_kernels = list(best_kernels_per_author.values())
        else:
            top_kernels = kernels[:top_k]

        for kernel in top_kernels:
            kernel_ref = kernel.ref
            self.api.kernels_pull(kernel_ref, path)
            logger.info(f"Downloaded kernel {kernel_ref} to {path}")


COMPETITIONS = {
    # classification
    "microsoft-malware-prediction": "AUC",
    "ieee-fraud-detection": "AUC",
    "santander-customer-satisfaction": "AUC",
    "talkingdata-adtracking-fraud-detection": "AUC",
    "porto-seguro-safe-driver-prediction": "Normalized Gini Coefficient",
    # regression
    "restaurant-revenue-prediction": "RMSE",
    "sberbank-russian-housing-market": "RMSLE",
    "ashrae-energy-prediction": "RMSLE",
    "mercedes-benz-greener-manufacturing": "R^2",
    # multi-class classification
    "prudential-life-insurance-assessment": "Quadratic Weighted Kappa",
    "otto-group-product-classification-challenge": "Logloss",
    "malware-classification": "Logloss",
    # time series
    "bike-sharing-demand": "RMSLE",
    "recruit-restaurant-visitor-forecasting": "RMSLE",
    "walmart-recruiting-store-sales-forecasting": "WMAE",
    "web-traffic-time-series-forecasting": "SMAPE",
}


def download_solutions():
    kaggle_client = KaggleClient()

    higher_metric = ["Normalized Gini Coefficient", "R^2", "Quadratic Weighted Kappa", "AUC", "F1", "Accuracy"]
    lower_metric = ["RMSE", "RMSLE", "Logloss", "WMAE", "SMAPE"]

    for k, v in COMPETITIONS.items():
        if v in higher_metric:
            kaggle_client.download_top_solution(
                competition=k, sort_by="scoreDescending"
            )
        elif v in lower_metric:
            kaggle_client.download_top_solution(competition=k, sort_by="scoreAscending")


if __name__ == "__main__":
    download_solutions()
