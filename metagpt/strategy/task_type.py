from enum import Enum

from pydantic import BaseModel

from metagpt.prompts.task_type import (
    DATA_PREPROCESS_PROMPT,
    EDA_PROMPT,
    FEATURE_ENGINEERING_PROMPT,
    IMAGE2WEBPAGE_PROMPT,
    MODEL_EVALUATE_PROMPT,
    MODEL_TRAIN_PROMPT,
    MODEL_ENSEMBLE_PROMPT,
)


class TaskTypeDef(BaseModel):
    name: str
    desc: str = ""
    guidance: str = ""


class TaskType(Enum):
    """By identifying specific types of tasks, we can inject human priors (guidance) to help task solving"""

    EDA = TaskTypeDef(
        name="eda",
        desc="For performing exploratory data analysis",
        guidance=EDA_PROMPT,
    )
    DATA_PREPROCESS = TaskTypeDef(
        name="data preprocessing",
        desc="Only for preprocessing the data before creating new features."
        "general data operation such as scaling and encoding never include.",
        guidance=DATA_PREPROCESS_PROMPT,
    )
    FEATURE_ENGINEERING = TaskTypeDef(
        name="feature engineering",
        desc="Only for creating new columns for input data, never include scaling or encoding.",
        guidance=FEATURE_ENGINEERING_PROMPT,
    )
    MODEL_PREPROCESS = TaskTypeDef(
        name="model preprocessing",
        desc="For preprocessing the data to ensure model training, such as scaling and encoding.",
        guidance="",
    )
    MODEL_TRAIN = TaskTypeDef(
        name="model train",
        desc="For training and evaluating model together in one task.",
        guidance=MODEL_TRAIN_PROMPT,
    )
    # MODEL_EVALUATE = TaskTypeDef(
    #     name="model evaluate",
    #     desc="Only for evaluating model.",
    #     guidance=MODEL_EVALUATE_PROMPT,
    # )
    MODEL_ENSEMBLE = TaskTypeDef(
        name="model ensemble",
        desc="Only for ensemble model.",
        guidance=MODEL_ENSEMBLE_PROMPT,
    )
    IMAGE2WEBPAGE = TaskTypeDef(
        name="image2webpage",
        desc="For converting image into webpage code.",
        guidance=IMAGE2WEBPAGE_PROMPT,
    )
    OTHER = TaskTypeDef(name="other", desc="Any tasks not in the defined categories")

    # Legacy TaskType to support tool recommendation using type match. You don't need to define task types if you have no human priors to inject.
    TEXT2IMAGE = TaskTypeDef(
        name="text2image",
        desc="Related to text2image, image2image using stable diffusion model.",
    )
    WEBSCRAPING = TaskTypeDef(
        name="web scraping",
        desc="For scraping data from web pages.",
    )
    EMAIL_LOGIN = TaskTypeDef(
        name="email login",
        desc="For logging to an email.",
    )

    @property
    def type_name(self):
        return self.value.name

    @classmethod
    def get_type(cls, type_name):
        for member in cls:
            if member.type_name == type_name:
                return member.value
        return None
