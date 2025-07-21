"""
This module implements data loaders for loading source data and managing data
preprocessing steps
"""
import pandas as pd
import numpy as np
import logging
import os
import json  # For metadata
import datetime as dt
import pandas.api.types as ptypes  # Allows O(1) check for datetime-like type

from pathlib import Path
from enum import Enum
from abc import ABC, abstractmethod
from text_preprocessing import TextPreprocessor
from author_disambiguation import AuthorDisambiguator
from dataframe_schema import DataColumn, COLUMN_TYPES
from utils import log_print


def get_project_root():
    """
    Returns the root directory one level up from mstml directory
    """
    # .parent gets directory containining this file
    # .parent.parent gets parent dir of dir containing this file
    return Path(__file__).resolve().parent.parent


def get_data_root():
    """
    Returns the data folder at the same level as the mstml directory
    """
    return get_project_root() / "data"


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
    def __init__(self, input_path: str, dataset_name: str, overwrite: bool = False) -> None:
        self.input_path = input_path
        self.dataset_name = dataset_name

        # Setup standard output directories
        self.output_dirs = self.setup_dataset_dirs(dataset_name, overwrite)
        self.output_dir = self.output_dirs["clean"]  # Default save location for main_df.pkl

        self.df = None  # Full dataframe
        self._valid_mask = None  # Boolean Series marking valid rows without NA

    @staticmethod
    def setup_dataset_dirs(dataset_name: str, overwrite: bool = False) -> dict:
        """
        Sets up standard folder structure under data/<dataset_name>:
        - original/
        - clean/
        - networks/

        Returns a dict of output paths.
        """
        data_root = get_data_root()
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


    @abstractmethod
    def load_raw_data(self):
        """Load data from the raw file(s). Should set self.raw_data"""
        pass


    @abstractmethod
    def preprocess(self):
        """
        Convert raw text corpus data into a DataFrame with standardized column
        format.

        Preprocessing capabilities are implemented in text_preprocessing.py

        Should set self.df
        """
        pass


    def _validate_and_flag(self):
        """
        Flags valid rows in self.df by checking for non-null required fields and correct types.
        Does not store multiple copies of the data.
        """
        if self.df is None:
            raise ValueError("self.df has not been populated.")

        required_cols = list(COLUMN_TYPES.keys())

        # Step 1: Initial mask: rows with no NA in required fields
        non_null_mask = self.df[required_cols].notna().all(axis=1)

        # Step 2: Refine mask based on type checks (fast via map for small types, ptypes for datetime)
        type_mask = non_null_mask.copy()
        for col, expected_type in COLUMN_TYPES.items():
            if expected_type is list:
                type_mask &= self.df[col].map(lambda x: isinstance(x, list))
            elif expected_type is str:
                type_mask &= self.df[col].map(lambda x: isinstance(x, str))
            elif expected_type is dt.date:
                # Use pandas datetime dtype check
                type_mask &= ptypes.is_datetime64_any_dtype(self.df[col])

        # Final mask
        self._valid_mask = type_mask

        # Logging results
        n_total = len(self.df)
        n_valid = self._valid_mask.sum()
        n_invalid = n_total - n_valid
        logging.info(f"Validation complete: {n_valid}/{n_total} rows valid, {n_invalid} invalid.")


    def save_outputs(self):
        """Save the cleaned DataFrame, metadata, and any additional variables."""
        df_path = os.path.join(self.output_dirs["clean"], 'main_df.pkl')
        self.df.to_pickle(df_path)
        logging.info(f"Saved full dataframe to: {df_path}")

        # Save clean and NA views for inspection
        self.get_clean_df().to_pickle(os.path.join(self.output_dirs["clean"], 'clean_df.pkl'))
        self.get_na_df().to_pickle(os.path.join(self.output_dirs["clean"], 'na_df.pkl'))

        metadata = self.generate_metadata()
        metadata_path = os.path.join(self.output_dirs["clean"], 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logging.info(f"Saved metadata to: {metadata_path}")

        # TODO: Save additional variables here


    def generate_metadata(self) -> dict:
        """Generate summary stats about the dataset. Override in subclass if needed."""
        if self._valid_mask is None:
            raise RuntimeError("Validation not run yet.")

        return {
            "input_path": self.input_path,
            "dataset_dir": self.output_dirs["root"],
            "total_rows": len(self.df),
            "valid_rows": int(self._valid_mask.sum()),
            "invalid_rows": int((~self._valid_mask).sum()),
            "date_range": {
                "min": str(self.df[DataColumn.DATE].min()),
                "max": str(self.df[DataColumn.DATE].max())
            }
        }


    def get_clean_df(self):
        """Get dataframe with only rows that have complete data"""
        if self._valid_mask is None:
            raise RuntimeError("Validation not run yet.")
        return self.df[self._valid_mask].sort_values(by=DataColumn.DATE)


    def get_na_df(self):
        """Get dataframe with only rows that are missing data"""
        if self._valid_mask is None:
            raise RuntimeError("Validation not run yet.")
        return self.df[~self._valid_mask].sort_values(by=DataColumn.DATE)


    def run(self):
        """Main pipeline: load, preprocess, validate, save"""
        self.load_raw_data()
        self.preprocess()
        self._validate_and_flag()
        self.save_outputs()
