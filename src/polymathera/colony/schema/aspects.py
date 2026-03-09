from enum import Enum
from pydantic import BaseModel

from .base_types import RepoId


class PredefinedCodeAspects(Enum):
    """
    Predefined code aspects that can be used to describe code bases in the UCR.
    """

    BUILD_SYSTEM = ("build_system",)
    TESTING = ("testing",)
    DOCUMENTATION = ("documentation",)
    DEPLOYMENT = ("deployment",)
    VERSION_CONTROL = ("version_control",)
    CONTAINERIZATION = ("containerization",)
    MONITORING = ("monitoring",)
    LOGGING = ("logging",)
    COMPUTE = ("compute",)
    STORAGE = ("storage",)
    NETWORKING = ("networking",)
    DATAFLOW = ("dataflow",)
    DISTRIBUTION = ("distribution",)
    MODEL_SERVING = ("model_serving",)
    MODEL_TRAINING = ("model_training",)
    VISUALIZATION = ("visualization",)
    LANGUAGE_BINDINGS = ("language_bindings",)
    SECURITY = ("security",)
    CONFIGURATION_MANAGEMENT = ("configuration_management",)
    PERFORMANCE = ("performance",)
    CI_CD = ("ci_cd",)
    IDE = ("ide",)


class CodeAspect(BaseModel):
    """
    Aspect-oriented programming (AOP) is used as a representational device to enable more
    concise descriptions of code in the prompts used in other analyses as well as retrieval
    and generation.

    AOP can also be viewed as a "reasoning device" since it allows for the abstraction of
    code into aspects that can help the LLM to focus its attention on the relevant parts of
    the code base.

    AOP can also be used to present a code base as a collection of "software stories" to
    help developers understand the code base better.

    # TODO: Abstraction: Support abstraction of aspects (parameterizing slices to make them reusable).

    An **abstracted program slice** (or a **view** or a **software story**) is a summary of
    a code base given a user query such that:
        - Code that is irrelevant to the query is abstracted using **black box constructs**
          either programmatic (e.g., abstract methods, classes and functions, etc.) or
          visual constructs (e.g., collapsible text, ellipsis, etc.).
        - Reconstruction of the original code base from the abstracted slice and its
        **complementary slice** is possible using **weaving instructions** (e.g., **aspects**).

        > A **complementary slice** is a slice that contains the code that was abstracted
        from the original slice.
    """

    repo: RepoId
    aspect_description: str
    source_map: dict[int, int]
    pointcut_list: list[str]
    advice_list: list[str]


# Alias for CodeAspect
CodeSlice = CodeAspect


class CodeSlicingCriteria(BaseModel):
    # TODO: One slicing criterion is all the code that is
    # reachable from the entry points invoked from another repository
    slicing_criteria: str
