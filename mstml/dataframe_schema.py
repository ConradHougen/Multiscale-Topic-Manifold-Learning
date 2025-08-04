"""
dataframe_schema.py

Defines schemas (column formats) for the pandas dataframes
"""

from enum import Enum
from collections import namedtuple
import datetime as dt

# Each schema column/field needs to defined in this format, with an extractor method
# The extractor method should perform basic minimal processing of the field
FieldDef = namedtuple("FieldDef", ["column_name", "extractor", "type"])

class MainDataSchema(Enum):
    """
    Defines the schemas used in pandas dataframes after loading data
    """
    TITLE = FieldDef(
        "title",
        lambda entry: entry.get("title", "").strip(),
        str
    )
    DATE = FieldDef(
        "date",
        lambda entry: (
            dt.date.fromisoformat(entry["date"][:10]) if entry.get("date") else None
        ),
        dt.date
    )
    RAW_TEXT = FieldDef(
        "raw_text",
        lambda entry: entry.get("raw_text", "").strip(),
        str
    )
    AUTHOR_NAMES = FieldDef(
        "author_names",
        lambda entry: (
            # Handle None authors
            [] if entry.get("authors") is None else
            # Handle list of dictionaries with "name" key
            [a["name"] for a in entry.get("authors", []) 
             if isinstance(a, dict) and "name" in a] if 
            any(isinstance(a, dict) for a in entry.get("authors", [])) else
            # Handle list of strings directly
            [str(a) for a in entry.get("authors", []) 
             if isinstance(a, str) and a.strip()]
        ),
        list
    )
    AUTHOR_IDS = FieldDef("author_ids", lambda entry: None, list)
    PREPROCESSED_TEXT = FieldDef("preprocessed_text", lambda entry: None, list)

    @property
    def colname(self):
        return self.value.column_name

    def get_extractor(self):
        return self.value.extractor

    @classmethod
    def all_colnames(cls):
        return [field.colname for field in cls]

    @classmethod
    def all_fields(cls):
        return list(cls)
