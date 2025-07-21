"""
dataframe_schema.py

Defines the schema (column format) for the pandas dataframe used for cleaned
text corpora.
"""

import datetime as dt
from enum import Enum

class DataColumn(str, Enum):
    """Defines the required and optional columns for the cleaned corpus df"""
    RAW_TEXT = "raw_text"
    TOKENIZED_TEXT = "tokenized_text"
    DATE = "date"
    AUTHOR_NAMES = "author_names"
    AUTHOR_IDS = "author_ids"

COLUMN_TYPES = {
    DataColumn.RAW_TEXT: str,
    DataColumn.TOKENIZED_TEXT: list,
    DataColumn.DATE: dt.date,
    DataColumn.AUTHOR_NAMES: list,
    DataColumn.AUTHOR_IDS: list,
}