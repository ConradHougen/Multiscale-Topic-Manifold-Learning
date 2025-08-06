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
            [] if entry.get("authors") is None else
            # Handle list of [first, last] pairs (ArXiv authors_parsed format)
            # Convert to nested list: [["LAST, FIRST"], ["LAST, FIRST"], ...]
            [[str.upper(str.strip(f"{auth[1]}, {auth[0]}"))] for auth in entry.get("authors", [])
             if isinstance(auth, (list, tuple)) and len(auth) >= 2] if
            isinstance(entry.get("authors"), list) and len(entry.get("authors", [])) > 0 and isinstance(entry.get("authors")[0], (list, tuple)) else
            
            # Handle semicolon-separated string format: "LAST, FIRST; LAST2, FIRST2"
            [[str.upper(str.strip(author))] for author in str(entry.get("authors", "")).split(";")
             if author.strip()] if isinstance(entry.get("authors"), str) and ";" in str(entry.get("authors", "")) else
            
            # Handle comma-separated string format (single author)
            [[str.upper(str.strip(str(entry.get("authors", ""))))]] if isinstance(entry.get("authors"), str) and entry.get("authors", "").strip() else
            
            # Handle flat list of strings (legacy format) - treat each as single author
            [[str.upper(str.strip(str(auth)))] for auth in entry.get("authors", [])
             if auth and str(auth).strip()] if isinstance(entry.get("authors"), list) else
            []
        ),
        list
    )
    AUTHOR_IDS = FieldDef("author_ids", lambda entry: None, list)
    PREPROCESSED_TEXT = FieldDef("preprocessed_text", lambda entry: None, list)
    CATEGORIES = FieldDef(
        "categories", 
        lambda entry: (
            entry.get("categories", "").split() if isinstance(entry.get("categories"), str) and entry.get("categories") else
            entry.get("categories", []) if isinstance(entry.get("categories"), list) else
            []
        ), 
        list
    )

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
