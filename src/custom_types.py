"""TODO: Module docstring."""

# Native
from enum import Enum


# Enum for possible classifications of words.
class ClassificationLabel(str, Enum):
    """TODO: Write Docstring."""

    ACCEPT = "accept"
    REJECT = "reject"
    UNSURE = "unsure"


# Enum for classification type/point.
class ClassificationType(str, Enum):
    """TODO: Write Docstring."""

    CLASSIFY = "classify"
    RECLASSIFY = "reclassify"
    REVIEW = "review"
