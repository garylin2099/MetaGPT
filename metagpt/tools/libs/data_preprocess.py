from __future__ import annotations

import json
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from metagpt.tools.tool_registry import register_tool

TAGS = ["data preprocessing", "machine learning"]


class MLProcess:
    def fit(self, df: pd.DataFrame):
        """
        Fit a model to be used in subsequent transform.

        Args:
            df (pd.DataFrame): The input DataFrame.
        """
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame with the fitted model.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the input DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        self.fit(df)
        return self.transform(df)


class DataPreprocessTool(MLProcess):
    """
    Completing a data preprocessing operation.
    """

    def __init__(self, features: list):
        """
        Initialize self.

        Args:
            features (list): Columns to be processed.
        """
        self.features = features
        self.model = None  # to be filled by specific subclass Tool

    def fit(self, df: pd.DataFrame):
        if len(self.features) == 0:
            return
        self.model.fit(df[self.features])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(self.features) == 0:
            return df
        new_df = df.copy()
        new_df[self.features] = self.model.transform(new_df[self.features])
        return new_df


@register_tool(tags=TAGS)
class FillMissingValue(DataPreprocessTool):
    """
    Completing missing values with simple strategies.
    """

    def __init__(
        self, features: list, strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean", fill_value=None
    ):
        """
        Initialize self.

        Args:
            features (list): Columns to be processed.
            strategy (Literal["mean", "median", "most_frequent", "constant"], optional): The imputation strategy, notice 'mean' and 'median' can only
                                      be used for numeric features. Defaults to 'mean'.
            fill_value (int, optional): Fill_value is used to replace all occurrences of missing_values.
                                        Defaults to None.
        """
        self.features = features
        self.model = SimpleImputer(strategy=strategy, fill_value=fill_value)


@register_tool(tags=TAGS)
class MinMaxScale(DataPreprocessTool):
    """
    Transform features by scaling each feature to a range, which is (0, 1).
    """

    def __init__(self, features: list):
        self.features = features
        self.model = MinMaxScaler()


@register_tool(tags=TAGS)
class StandardScale(DataPreprocessTool):
    """
    Standardize features by removing the mean and scaling to unit variance.
    """

    def __init__(self, features: list):
        self.features = features
        self.model = StandardScaler()


@register_tool(tags=TAGS)
class MaxAbsScale(DataPreprocessTool):
    """
    Scale each feature by its maximum absolute value.
    """

    def __init__(self, features: list):
        self.features = features
        self.model = MaxAbsScaler()


@register_tool(tags=TAGS)
class RobustScale(DataPreprocessTool):
    """
    Apply the RobustScaler to scale features using statistics that are robust to outliers.
    """

    def __init__(self, features: list):
        self.features = features
        self.model = RobustScaler()


@register_tool(tags=TAGS)
class OrdinalEncode(DataPreprocessTool):
    """
    Encode categorical features as ordinal integers.
    """

    def __init__(self, features: list):
        self.features = features
        self.model = OrdinalEncoder()


@register_tool(tags=TAGS)
class OneHotEncode(DataPreprocessTool):
    """
    Apply one-hot encoding to specified categorical columns, the original columns will be dropped.
    """

    def __init__(self, features: list):
        self.features = features
        self.model = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ts_data = self.model.transform(df[self.features])
        new_columns = self.model.get_feature_names_out(self.features)
        ts_data = pd.DataFrame(ts_data, columns=new_columns, index=df.index)
        new_df = df.drop(self.features, axis=1)
        new_df = pd.concat([new_df, ts_data], axis=1)
        return new_df


@register_tool(tags=TAGS)
class LabelEncode(DataPreprocessTool):
    """
    Apply label encoding to specified categorical columns in-place.
    """

    def __init__(self, features: list):
        """
        Initialize self.

        Args:
            features (list): Categorical columns to be label encoded.
        """
        self.features = features
        self.le_encoders = []

    def fit(self, df: pd.DataFrame):
        if len(self.features) == 0:
            return
        for col in self.features:
            le = LabelEncoder().fit(df[col].astype(str).unique().tolist() + ["unknown"])
            self.le_encoders.append(le)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(self.features) == 0:
            return df
        new_df = df.copy()
        for i in range(len(self.features)):
            data_list = df[self.features[i]].astype(str).tolist()
            for unique_item in np.unique(df[self.features[i]].astype(str)):
                if unique_item not in self.le_encoders[i].classes_:
                    data_list = ["unknown" if x == unique_item else x for x in data_list]
            new_df[self.features[i]] = self.le_encoders[i].transform(data_list)
        return new_df


def get_column_info(df: pd.DataFrame, max_cols=10) -> str:
    """
    Analyzes a DataFrame and categorizes its columns based on data types.

    Args:
        df (pd.DataFrame): The DataFrame to be analyzed.
        max_cols (int, optional): The maximum number of columns to show for each category. Defaults to 10.

    Returns:
        str: The formatted column info.
    """
    column_info = {
        "Category": [],
        "Numeric": [],
        "Datetime": [],
        "Other": [],
    }
    for col in df.columns:
        data_type = str(df[col].dtype).replace("dtype('", "").replace("')", "")
        if data_type.startswith("object"):
            column_info["Category"].append(col)
        elif data_type.startswith("int") or data_type.startswith("float"):
            column_info["Numeric"].append(col)
        elif data_type.startswith("datetime"):
            column_info["Datetime"].append(col)
        else:
            column_info["Other"].append(col)

    result = []
    for key, value in column_info.items():
        col_count = len(value)
        if col_count > max_cols:
            displayed_cols = value[:max_cols]
            result.append(f"{key} Columns (Only show {max_cols} of {col_count}): {', '.join(displayed_cols)}, ...")
        else:
            result.append(f"{key} Columns ({col_count} columns): {', '.join(value)}")

    info = f"Number of Rows: {len(df)}\nNumber of Columns: {len(df.columns)}\n\n" + "\n".join(result)
    return info


def get_data_info(train_path, target):
    train_data = pd.read_csv(train_path)
    num_rows, num_cols = train_data.shape
    missing_values_info = train_data.isnull().sum() / len(train_data) * 100

    numeric_features = train_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = train_data.select_dtypes(include=[object]).columns.tolist()

    unique_values = train_data[target].nunique()

    if unique_values <= 10:
        target_info = train_data[target].value_counts().to_dict()
        target_info_str = "Value Counts:\n" + "\n".join([f"{val}: {count}" for val, count in target_info.items()])
    else:
        target_info = train_data[target].describe().to_dict()
        target_info_str = "Description:\n" + "\n".join([f"{key}: {val}" for key, val in target_info.items()])

    if target in numeric_features:
        numeric_features.remove(target)
    if target in categorical_features:
        categorical_features.remove(target)

    display_columns = 10
    numeric_features_str = "\n".join(
        [f"{col:<20} {missing_values_info[col]:.2f}%" for col in numeric_features[:display_columns]])

    categorical_features_info = {}
    for col in categorical_features[:display_columns]:
        unique_values = train_data[col].nunique()
        top_values = train_data[col].value_counts().head(5).to_dict()
        top_values_str = ", ".join([f"{k}: {v}" for k, v in top_values.items()])
        categorical_features_info[col] = {
            'missing_rate': missing_values_info[col],
            'unique_values': unique_values,
            'top_values': top_values_str
        }

    categorical_features_str = "\n".join([
        f"{col:<20} {categorical_features_info[col]['missing_rate']:>6.2f}%    {categorical_features_info[col]['unique_values']:<10} {categorical_features_info[col]['top_values']}"
        for col in categorical_features[:display_columns]
    ])

    data_info = f"""
## Basic Info:
Number of Rows: {num_rows}
Number of Columns: {num_cols}

## Numeric Features (Only show first {min(display_columns, len(numeric_features))} of {len(numeric_features)}):
Column_Name        Missing_Rate
{numeric_features_str}
{"..." if len(numeric_features) > display_columns else ""}

## Categorical Features (Only show first {min(display_columns, len(categorical_features))} of {len(categorical_features)}):
Column_Name        Missing_Rate     N_unique   Top5_Value_Counts
{categorical_features_str}
{"..." if len(categorical_features) > display_columns else ""}

## Target ({target}) Info:
{target_info_str}
"""

    return data_info
