"""
This module implements data loaders for loading source data and managing data
preprocessing steps
"""
import pandas as pd
import logging
import os
import json  # For metadata
import datetime as dt
import pandas.api.types as ptypes  # Allows O(1) check for datetime-like type

from typing import Optional, Dict
from pathlib import Path
from abc import ABC, abstractmethod

from text_preprocessing import TextPreprocessor
from author_disambiguation import AuthorDisambiguator
from dataframe_schema import MainDataSchema
from utils import log_print, validate_dataset_name


def get_project_root_directory():
    """
    Returns the root directory one level up from mstml directory
    """
    # .parent gets directory containing this file
    # .parent.parent gets parent dir of dir containing this file
    return Path(__file__).resolve().parent.parent


def get_data_directory():
    """
    Returns the data folder at the same level as the mstml directory
    """
    return get_project_root_directory() / "data"

"""============================================================================
class DataLoader(ABC)

This class is the base class for data loaders. Data loaders help parse source
data into cleaned pandas dataframes with a particular schema, as specified by
dataframe_schema.py.

Basic Usage: 
    1. Create a DataLoader instance, given an input_path and valid dataset_name
    2. overwrite parameter can be used to force clobber data.
    3. Use .run() method for creating and populating directories in /data/
    4. /data/ should contain original data, clean data, and networks
============================================================================"""
class DataLoader(ABC):
    """
    Base class for data loaders; This class should be extended to enable
    loading from multiple types of source documents, including but not
    limited to:

    1. JSON files
    2. Web APIs
    3. CSV files
    4. TXT files
    5. Pandas DataFrames without correct formatting
    6. Zip files
    7. A local directory with multiple files and formats
    """
    def __init__(self,
                 input_path: str,
                 dataset_name: str,
                 overwrite: bool = False,
                 text_preprocessor: Optional[TextPreprocessor] = None,
                 author_disambiguator: Optional[AuthorDisambiguator] = None,
                 input_schema_map: Optional[Dict[str, str]] = None) -> None:
        # Ensure that dataset_name is coherent for a directory name
        validate_dataset_name(dataset_name)

        # Copy input to internal variables
        self._original_input_path = Path(input_path)
        self._dataset_name = dataset_name
        self._overwrite = overwrite
        self.text_preprocessor = text_preprocessor or TextPreprocessor()
        self.author_disambiguator = author_disambiguator or AuthorDisambiguator()
        self.input_schema_map = input_schema_map or {}

        # To be resolved later
        self.dataset_dirs = None
        self.input_path = None
        self.df = None  # Full dataframe
        self._valid_mask = None  # Boolean Series marking valid rows without NA

    def get_clean_df(self):
        """Get dataframe with only rows that have complete data (no NA)"""
        if self._valid_mask is None:
            raise RuntimeError("Validation not run yet.")
        return self.df[self._valid_mask].sort_values(by=MainDataSchema.DATE)

    def get_na_df(self):
        """Get dataframe with only rows that are missing data (NA entries)"""
        if self._valid_mask is None:
            raise RuntimeError("Validation not run yet.")
        return self.df[~self._valid_mask].sort_values(by=MainDataSchema.DATE)

    def run(self):
        """Main pipeline: prepare, load, preprocess, validate, save"""
        self._prepare_environment()
        self._prepare_input()
        self._load_raw_data()
        self._preprocess()
        self._validate_and_flag()
        self._save_outputs()

    @property
    def dataset_root_dir(self):
        return self.dataset_dirs["root"]

    @property
    def dataset_original_dir(self):
        return self.dataset_dirs["original"]

    @property
    def dataset_clean_dir(self):
        return self.dataset_dirs["clean"]

    @property
    def dataset_networks_dir(self):
        return self.dataset_dirs["clean"]

    @staticmethod
    def setup_dataset_dirs(dataset_name: str, overwrite: bool = False) -> dict:
        """
        Sets up standard folder structure under data/<dataset_name>:
        - original/
        - clean/
        - networks/

        Returns a dict of output paths.
        """
        data_root = get_data_directory()
        setup_msg = f"Setting up data directory for {dataset_name} at {data_root}..."
        log_print(setup_msg)
        dataset_dir = data_root / dataset_name
        original_dir = dataset_dir / "original"
        clean_dir = dataset_dir / "clean"
        networks_dir = dataset_dir / "networks"

        for subdir in [original_dir, clean_dir, networks_dir]:
            if subdir.exists() and not overwrite:
                warn_msg = f"{subdir} already exists. Use `overwrite=True` to recreate it."
                log_print(warn_msg, level="warning")
            else:
                subdir.mkdir(parents=True, exist_ok=True)

        return {
            "root": dataset_dir,
            "original": original_dir,
            "clean": clean_dir,
            "networks": networks_dir,
        }

    """=====================================================================
    PRIVATE METHODS
    ========================================================================"""

    @abstractmethod
    def _load_raw_data(self):
        """Load data from the raw file(s). Should set self.raw_data"""
        pass

    @abstractmethod
    def _preprocess(self):
        """
        Convert raw text corpus data into a DataFrame with standardized column
        format.

        Preprocessing capabilities are implemented in text_preprocessing.py

        Should set self.df
        """
        pass

    def _prepare_environment(self):
        """
        Creates the dataset folder structure and sets output_dirs.
        """
        self.dataset_dirs = self.setup_dataset_dirs(self._dataset_name, self._overwrite)

    def _prepare_input(self):
        """
        Resolves or downloads input_path into the original folder.
        Updates self.input_path.
        """
        self.input_path = self._resolve_input_path(self._original_input_path)

    def _resolve_input_path(self, input_path: Path) -> Path:
        """
        Ensures the input file is located under original/, copying it there if needed.
        If it's already in original/, do nothing.
        If it's a URL, download it.
        """
        original_dir = self.dataset_original_dir
        input_path = input_path.expanduser().resolve()

        # Case 1: input is already under original_dir
        if original_dir in input_path.parents:
            log_print(f"Input file already in correct location: {input_path}")
            return input_path

        # Case 2: input is a URL
        if input_path.as_posix().startswith("http://") or input_path.as_posix().startswith("https://"):
            filename = input_path.name
            target_path = original_dir / filename
            if not target_path.exists():
                import requests
                log_print(f"Downloading from URL: {input_path}")
                response = requests.get(input_path.as_posix())
                response.raise_for_status()
                with open(target_path, "wb") as f:
                    f.write(response.content)
                log_print(f"Downloaded to: {target_path}")
            return target_path

        # Case 3: input is a local file outside the expected directory — copy it
        if input_path.is_file():
            target_path = original_dir / input_path.name
            if not target_path.exists():
                log_print(f"Copying {input_path} → {target_path}")
                import shutil
                shutil.copy(input_path, target_path)
            return target_path

        raise FileNotFoundError(f"Input path '{input_path}' not found or unsupported.")

    def _validate_and_flag(self):
        """
        Flags valid rows in self.df by checking for non-null required fields and correct types.
        Does not store multiple copies of the data.
        """
        if self.df is None:
            raise ValueError("self.df has not been populated.")

        # Get all column names and types from the schema
        schema_fields = MainDataSchema.all_fields()

        required_cols = [field.colname for field in schema_fields]
        col_type_map = {field.colname: field.value.type for field in schema_fields}

        # Step 1: Initial mask: rows with no NA in required fields
        non_null_mask = self.df[required_cols].notna().all(axis=1)

        # General-purpose type checking function
        def check_type(col: pd.Series, expected_type: type) -> pd.Series:
            if expected_type is list or expected_type is str:
                return col.map(lambda x: isinstance(x, expected_type))
            elif expected_type is dt.date:
                # If the column is already datetime64, assume valid
                if ptypes.is_datetime64_any_dtype(col):
                    return pd.Series(True, index=col.index)
                # Otherwise, fallback to row-wise isinstance check
                return col.map(lambda x: isinstance(x, dt.date))
            else:
                # General fallback for other types
                return col.map(lambda x: isinstance(x, expected_type))

        type_mask = non_null_mask.copy()
        for col, expected_type in col_type_map.items():
            try:
                result = check_type(self.df[col], expected_type)
                if isinstance(result, bool):
                    # Some checks return a scalar (e.g., is_datetime64_any_dtype); broadcast it
                    result = pd.Series([result] * len(self.df), index=self.df.index)
                type_mask &= result
            except Exception as e:
                logging.warning(f"Type check for column '{col}' failed: {e}")
                type_mask &= False  # Fail closed

        # Final mask
        self._valid_mask = type_mask

        # Logging results
        n_total = len(self.df)
        n_valid = self._valid_mask.sum()
        n_invalid = n_total - n_valid
        logging.info(f"Validation complete: {n_valid}/{n_total} rows valid, {n_invalid} invalid.")

    def _save_outputs(self):
        """Save the cleaned DataFrame, metadata, and any additional variables."""
        df_path = os.path.join(self.dataset_dirs["clean"], 'main_df.pkl')
        self.df.to_pickle(df_path)
        logging.info(f"Saved full dataframe to: {df_path}")

        # Save clean and NA views for inspection
        self.get_clean_df().to_pickle(os.path.join(self.dataset_dirs["clean"], 'clean_df.pkl'))
        self.get_na_df().to_pickle(os.path.join(self.dataset_dirs["clean"], 'na_df.pkl'))

        metadata = self._generate_metadata()
        metadata_path = os.path.join(self.dataset_dirs["clean"], 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logging.info(f"Saved metadata to: {metadata_path}")

        # Save additional variables as needed

    def _generate_metadata(self) -> dict:
        """Generate summary stats about the dataset. Override in subclass if needed."""
        if self._valid_mask is None:
            raise RuntimeError("Validation not run yet.")

        return {
            "input_path": self.input_path,
            "dataset_dir": self.dataset_dirs["root"],
            "total_rows": len(self.df),
            "valid_rows": int(self._valid_mask.sum()),
            "invalid_rows": int((~self._valid_mask).sum()),
            "date_range": {
                "min": str(self.df[MainDataSchema.DATE].min()),
                "max": str(self.df[MainDataSchema.DATE].max())
            }
        }

    def _apply_preprocessors(self):
        """
        Fills in author IDs and preprocessed text using their respective modules.
        Assumes author_disambiguator and text_preprocessor are initialized.
        """
        if self.df is None or self.df.empty:
            raise ValueError("Dataframe is empty. Cannot apply preprocessors.")

        # Preprocess text
        if self.text_preprocessor:
            try:
                self.df[MainDataSchema.PREPROCESSED_TEXT.colname] = self.text_preprocessor.update_dataframe(
                    self.df, MainDataSchema.RAW_TEXT.colname
                )
            except Exception as e:
                logging.warning(f"Text preprocessing failed: {e}")

        # Disambiguate authors
        if self.author_disambiguator:
            try:
                self.df[MainDataSchema.AUTHOR_IDS.colname] = self.author_disambiguator.update_dataframe(
                    self.df, MainDataSchema.AUTHOR_NAMES.colname
                )
            except Exception as e:
                logging.warning(f"Author disambiguation failed: {e}")


"""============================================================================
class JsonDataLoader(DataLoader)

This class implements a DataLoader for a JSON-formatted text corpus. The path
to a valid JSON file with one document per entry should be passed as 
input_path. The data will be loaded and preprocessed accordingly.

============================================================================"""
class JsonDataLoader(DataLoader):
    def _load_raw_data(self):
        """
        Loads JSON lines file (one JSON object per line).
        Sets self.raw_data to a list of parsed entries.
        """
        with open(self.input_path, 'r', encoding='utf-8') as f:
            self.raw_data = [json.loads(line) for line in f]
        logging.info(f"Loaded {len(self.raw_data)} entries from {self.input_path}")

    def _preprocess(self):
        """
        Converts each raw entry into a row conforming to MainDataSchema using extractors.
        Then applies additional preprocessing for text and authors.
        """
        rows = []
        for entry in self.raw_data:
            try:
                row = {
                    field.colname: field.get_extractor()(entry)
                    for field in MainDataSchema
                }
                rows.append(row)
            except Exception as e:
                logging.warning(f"Skipping malformed entry due to error: {e}")

        self.df = pd.DataFrame(rows)
        self._apply_preprocessors()
