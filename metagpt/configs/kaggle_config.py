#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/12 14:15
# @Author  : lidanyang
# @File    : kaggle_config.py
# @Desc    :
import os

from metagpt.utils.yaml_model import YamlModelWithoutDefault


class KaggleConfig(YamlModelWithoutDefault):
    username: str
    key: str

    def setup_api_credentials(self):
        """Setup Kaggle API credentials and authenticate"""
        os.environ["KAGGLE_USERNAME"] = self.username
        os.environ["KAGGLE_KEY"] = self.key
