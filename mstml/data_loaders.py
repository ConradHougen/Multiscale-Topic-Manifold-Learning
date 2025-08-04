"""
This module implements data loaders for loading source data and managing data
preprocessing steps
"""
import pandas as pd
import os
import json  # For metadata
import datetime as dt
import pandas.api.types as ptypes  # Allows O(1) check for datetime-like type

from typing import Optional, Dict, Union, List
from pathlib import Path
from abc import ABC, abstractmethod

from .author_disambiguation import AuthorDisambiguator
from .dataframe_schema import MainDataSchema
from ._file_driver import log_print, validate_dataset_name, write_pickle
from .data_loader_registry import DataLoaderRegistry, register_data_loader


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

def get_dataset_directory(dataset_name: str):
    """
    Returns the directory specific to the given dataset name
    """
    return get_data_directory() / dataset_name

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
                 input_path: Union[str, List[str]],
                 dataset_name: str,
                 overwrite: bool = False,
                 author_disambiguator: Optional[AuthorDisambiguator] = None,
                 input_schema_map: Optional[Dict[str, str]] = None,
                 data_filters: Optional[Dict[str, any]] = None) -> None:
        # Ensure that dataset_name is coherent for a directory name
        validate_dataset_name(dataset_name)

        # Normalize input to list of paths
        if isinstance(input_path, str):
            self._original_input_paths = [Path(input_path)]
        else:
            self._original_input_paths = [Path(p) for p in input_path]
        
        # Copy input to internal variables
        self._dataset_name = dataset_name
        self._overwrite = overwrite
        self.author_disambiguator = author_disambiguator or AuthorDisambiguator()
        self.input_schema_map = input_schema_map or {}
        self.data_filters = data_filters or {}

        # To be resolved later
        self.dataset_dirs = None
        self.input_paths = None  # Now a list of resolved paths
        self.df = None  # Full dataframe
        self._valid_mask = None  # Boolean Series marking valid rows without NA

    def get_clean_df(self):
        """Get dataframe with only rows that have complete data (no NA)"""
        if self._valid_mask is None:
            raise RuntimeError("Validation not run yet.")
        return self.df[self._valid_mask].sort_values(by=MainDataSchema.DATE.colname)

    def get_na_df(self):
        """Get dataframe with only rows that are missing data (NA entries)"""
        if self._valid_mask is None:
            raise RuntimeError("Validation not run yet.")
        return self.df[~self._valid_mask].sort_values(by=MainDataSchema.DATE.colname)

    def run(self):
        """Main pipeline: prepare, load, preprocess, filter, validate, save"""
        self._prepare_environment()
        self._prepare_input()
        self._load_raw_data()
        self._preprocess()  # Basic preprocessing only
        self._apply_data_filters()  # Filter data by categories, dates, etc.
        self._apply_author_disambiguation()  # Author disambiguation on reduced df for speed
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
        return self.dataset_dirs["networks"]

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
        Resolves or downloads input_paths into the original folder.
        Updates self.input_paths.
        """
        self.input_paths = [self._resolve_input_path(path) for path in self._original_input_paths]

    def _resolve_input_path(self, input_path: Path) -> Path:
        """
        Ensures the input file is located under original/, copying it there if needed.
        Handles multiple cases:
        1. File already in original/ directory
        2. Just a filename - look for it in original/ directory
        3. Absolute path to existing file - copy to original/
        4. URL - download to original/
        """
        original_dir = self.dataset_original_dir
        
        # Case 1: Check if it's just a filename and exists in original directory
        if not input_path.is_absolute():
            # It's a relative path or just a filename
            target_path = original_dir / input_path.name
            if target_path.exists():
                log_print(f"Found file in dataset original directory: {target_path}")
                return target_path
        
        # Expand and resolve the path for absolute path handling
        resolved_input_path = input_path.expanduser().resolve()

        # Case 2: input is already under original_dir
        if original_dir in resolved_input_path.parents or resolved_input_path.parent == original_dir:
            log_print(f"Input file already in correct location: {resolved_input_path}")
            return resolved_input_path

        # Case 3: input is a URL
        input_str = str(input_path)
        if input_str.startswith("http://") or input_str.startswith("https://"):
            filename = input_path.name
            target_path = original_dir / filename
            if not target_path.exists():
                import requests
                log_print(f"Downloading from URL: {input_path}")
                response = requests.get(input_str)
                response.raise_for_status()
                with open(target_path, "wb") as f:
                    f.write(response.content)
                log_print(f"Downloaded to: {target_path}")
            return target_path

        # Case 4: input is a local file outside the expected directory — copy it
        if resolved_input_path.is_file():
            target_path = original_dir / resolved_input_path.name
            if not target_path.exists():
                log_print(f"Copying {resolved_input_path} → {target_path}")
                import shutil
                shutil.copy(resolved_input_path, target_path)
            return target_path

        # Case 5: File not found anywhere
        # Provide helpful error message with suggestions
        target_path = original_dir / input_path.name
        available_files = [f.name for f in original_dir.iterdir() if f.is_file()] if original_dir.exists() else []
        
        error_msg = f"Input file '{input_path}' not found.\\n"
        error_msg += f"Looked for: {target_path}\\n"
        if available_files:
            error_msg += f"Available files in {original_dir}: {available_files}"
        else:
            error_msg += f"No files found in {original_dir} (directory may not exist)"
        
        raise FileNotFoundError(error_msg)

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
                log_print(f"Type check for column '{col}' failed: {e}", level="warning")
                type_mask &= False  # Fail closed

        # Final mask
        self._valid_mask = type_mask

        # Logging results
        n_total = len(self.df)
        n_valid = self._valid_mask.sum()
        n_invalid = n_total - n_valid
        log_print(f"Validation complete: {n_valid}/{n_total} rows valid, {n_invalid} invalid.", level="info")

    def _apply_author_disambiguation(self):
        """
        Apply author disambiguation after data filtering to improve performance.
        """
        if self.author_disambiguator:
            try:
                author_ids_col = MainDataSchema.AUTHOR_IDS.colname
                author_names_col = MainDataSchema.AUTHOR_NAMES.colname
                
                if author_names_col in self.df.columns:
                    self.df[author_ids_col] = self.author_disambiguator.update_dataframe(
                        self.df, author_names_col
                    )
                    log_print("Applied author disambiguation", level="info")
                else:
                    log_print(f"Column '{author_names_col}' not found - skipping author disambiguation", level="warning")
            except Exception as e:
                log_print(f"Author disambiguation failed: {e}", level="warning")
        else:
            log_print("No author disambiguator provided - using author names as IDs", level="info")
    
    def _apply_data_filters(self):
        """
        Apply data filters to reduce dataset size before expensive operations.
        
        Filters are applied in the order they appear in self.data_filters dict.
        This step occurs after preprocessing but before author disambiguation
        and validation to improve performance.
        """
        if not self.data_filters or self.df is None or self.df.empty:
            return
        
        original_count = len(self.df)
        log_print(f"Applying data filters to {original_count} rows...", level="info")
        
        for filter_name, filter_config in self.data_filters.items():
            if filter_config is None:
                continue
                
            try:
                self.df = self._apply_single_filter(filter_name, filter_config, self.df)
                current_count = len(self.df)
                log_print(f"After {filter_name} filter: {current_count} rows ({original_count - current_count} removed)", level="info")
                
                if self.df.empty:
                    log_print("Warning: All rows filtered out. Consider relaxing filter criteria.", level="warning")
                    break
                    
            except Exception as e:
                log_print(f"Error applying {filter_name} filter: {e}", level="error")
                continue
        
        final_count = len(self.df)
        if final_count < original_count:
            reduction_pct = ((original_count - final_count) / original_count) * 100
            log_print(f"Data filtering complete: {final_count}/{original_count} rows retained ({reduction_pct:.1f}% reduction)", level="info")
    
    def _apply_single_filter(self, filter_name: str, filter_config: any, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a single filter to the dataframe.
        
        Args:
            filter_name: Name of the filter for logging
            filter_config: Filter configuration (format depends on filter type)
            df: DataFrame to filter
            
        Returns:
            Filtered DataFrame
        """
        if filter_name == 'date_range':
            return self._apply_date_range_filter(filter_config, df)
        elif filter_name == 'categories':
            return self._apply_categories_filter(filter_config, df)
        elif filter_name == 'custom':
            return self._apply_custom_filter(filter_config, df)
        else:
            log_print(f"Unknown filter type: {filter_name}", level="warning")
            return df
    
    def _apply_date_range_filter(self, date_config: Dict[str, str], df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter by date range.
        
        Args:
            date_config: Dict with 'start' and/or 'end' keys (ISO date strings)
            df: DataFrame to filter
            
        Returns:
            Filtered DataFrame
        """
        date_col = MainDataSchema.DATE.colname
        
        if date_col not in df.columns:
            log_print(f"Date column '{date_col}' not found, skipping date filter", level="warning")
            return df
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        mask = pd.Series(True, index=df.index)
        
        if 'start' in date_config and date_config['start']:
            start_date = pd.to_datetime(date_config['start'])
            mask &= (df[date_col] >= start_date)
            log_print(f"   Date filter: >= {start_date.date()}", level="debug")
        
        if 'end' in date_config and date_config['end']:
            end_date = pd.to_datetime(date_config['end'])
            mask &= (df[date_col] <= end_date)
            log_print(f"   Date filter: <= {end_date.date()}", level="debug")
        
        return df[mask].copy()
    
    def _apply_categories_filter(self, categories_config: Union[List[str], Dict], df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter by categories (for arXiv-style data).
        
        Args:
            categories_config: List of category labels to include, or dict with config
            df: DataFrame to filter
            
        Returns:
            Filtered DataFrame
        """
        # Handle different config formats
        if isinstance(categories_config, list):
            target_categories = categories_config
            column = 'categories'  # Default column name
        elif isinstance(categories_config, dict):
            target_categories = categories_config.get('include', [])
            column = categories_config.get('column', 'categories')
        else:
            log_print(f"Invalid categories filter config: {categories_config}", level="warning")
            return df
        
        if not target_categories:
            log_print("No target categories specified, skipping categories filter", level="warning")
            return df
        
        if column not in df.columns:
            log_print(f"Categories column '{column}' not found, skipping categories filter", level="warning")
            return df
        
        log_print(f"   Categories filter: looking for {target_categories} in column '{column}'", level="debug")
        
        # Apply the category filter (similar to your original implementation)
        def check_categories(cat_value):
            if pd.isna(cat_value) or not cat_value:
                return False
            # Convert to string and check if any target category is in the value
            cat_str = str(cat_value)
            return any(cat in cat_str for cat in target_categories)
        
        cat_mask = df[column].apply(check_categories)
        return df[cat_mask].copy()
    
    def _apply_custom_filter(self, custom_config: Dict, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a custom filter using a lambda function or callable.
        
        Args:
            custom_config: Dict with 'function' key containing a callable
            df: DataFrame to filter
            
        Returns:
            Filtered DataFrame
        """
        if not isinstance(custom_config, dict) or 'function' not in custom_config:
            log_print("Custom filter requires a 'function' key with a callable", level="warning")
            return df
        
        filter_func = custom_config['function']
        if not callable(filter_func):
            log_print("Custom filter 'function' must be callable", level="warning")
            return df
        
        try:
            mask = filter_func(df)
            return df[mask].copy()
        except Exception as e:
            log_print(f"Custom filter function failed: {e}", level="error")
            return df

    def _save_outputs(self):
        """Save the cleaned DataFrame, metadata, and any additional variables."""
        import pickle
        from collections import defaultdict
        
        df_path = os.path.join(self.dataset_dirs["clean"], 'main_df.pkl')
        write_pickle(df_path, self.df)
        log_print(f"Saved full dataframe to: {df_path}", level="info")

        # Save clean and NA views for inspection
        write_pickle(os.path.join(self.dataset_dirs["clean"], 'clean_df.pkl'), self.get_clean_df())
        write_pickle(os.path.join(self.dataset_dirs["clean"], 'na_df.pkl'), self.get_na_df())

        # Generate and save author mapping dictionaries
        self._save_author_mappings()
        
        # Note: Vocabulary dictionary (id2word.pkl) now generated by MstmlOrchestrator during text preprocessing

        metadata = self._generate_metadata()
        metadata_path = os.path.join(self.dataset_dirs["clean"], 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        log_print(f"Saved metadata to: {metadata_path}", level="info")

    def _save_author_mappings(self):
        """Generate and save author mapping dictionaries."""
        from collections import defaultdict
        from .dataframe_schema import MainDataSchema
        
        # Create author mappings
        authorId_to_df_row = defaultdict(list)
        author_to_authorId = defaultdict(set)  # Use set to handle duplicates
        authorId_to_author = {}
        
        for idx, row in self.df.iterrows():
            author_names = row.get(MainDataSchema.AUTHOR_NAMES.colname, [])
            author_ids = row.get(MainDataSchema.AUTHOR_IDS.colname, [])
            
            # Handle None values gracefully
            if author_names is None:
                author_names = []
            if author_ids is None:
                author_ids = []
            
            # Ensure we have lists
            if isinstance(author_names, str):
                author_names = [author_names]
            if isinstance(author_ids, str):
                author_ids = [author_ids]
                
            # Handle case where we have names but no IDs (use names as IDs)
            if author_names and not author_ids:
                author_ids = author_names
            
            # Skip rows with no author information
            if not author_ids:
                continue
            
            # Build mappings
            for i, author_id in enumerate(author_ids):
                author_id = str(author_id)
                
                # Map author ID to dataframe row
                authorId_to_df_row[author_id].append(idx)
                
                # Get corresponding author name
                if i < len(author_names):
                    author_name = str(author_names[i])
                else:
                    author_name = author_id  # Fallback to ID as name
                
                # Map author name to author ID
                author_to_authorId[author_name].add(author_id)
                
                # Map author ID to author name
                authorId_to_author[author_id] = author_name
        
        # Convert sets to lists for serialization
        author_to_authorId = {name: list(ids) for name, ids in author_to_authorId.items()}
        
        # Save the mappings
        clean_dir = self.dataset_dirs["clean"]
        
        write_pickle(os.path.join(clean_dir, 'authorId_to_df_row.pkl'), dict(authorId_to_df_row))
        log_print("Saved authorId_to_df_row.pkl", level="info")
        
        write_pickle(os.path.join(clean_dir, 'author_to_authorId.pkl'), author_to_authorId)
        log_print("Saved author_to_authorId.pkl", level="info")
        
        write_pickle(os.path.join(clean_dir, 'authorId_to_author.pkl'), authorId_to_author)
        log_print("Saved authorId_to_author.pkl", level="info")
    

    def _generate_metadata(self) -> dict:
        """Generate summary stats about the dataset. Override in subclass if needed."""
        if self._valid_mask is None:
            raise RuntimeError("Validation not run yet.")

        return {
            "input_paths": [str(path) for path in self.input_paths],
            "dataset_dir": self.dataset_dirs["root"],
            "total_rows": len(self.df),
            "valid_rows": int(self._valid_mask.sum()),
            "invalid_rows": int((~self._valid_mask).sum()),
            "date_range": {
                "min": str(self.df[MainDataSchema.DATE.colname].min()),
                "max": str(self.df[MainDataSchema.DATE.colname].max())
            }
        }


"""============================================================================
class JsonDataLoader(DataLoader)

This class implements a DataLoader for a JSON-formatted text corpus. The path
to a valid JSON file with one document per entry should be passed as 
input_path. The data will be loaded and preprocessed accordingly.

============================================================================"""
@register_data_loader('json', 'jsonl')
class JsonDataLoader(DataLoader):
    def _load_raw_data(self):
        """
        Loads JSON lines file(s) (one JSON object per line).
        If multiple files are provided, concatenates all entries.
        Sets self.raw_data to a list of parsed entries.
        """
        self.raw_data = []
        total_entries = 0
        
        for input_path in self.input_paths:
            with open(input_path, 'r', encoding='utf-8') as f:
                file_data = [json.loads(line) for line in f]
                self.raw_data.extend(file_data)
                file_count = len(file_data)
                total_entries += file_count
                log_print(f"Loaded {file_count} entries from {input_path}", level="info")
        
        log_print(f"Total loaded: {total_entries} entries from {len(self.input_paths)} file(s)", level="info")

    def _preprocess(self):
        """
        Converts each raw entry into a row conforming to MainDataSchema using extractors.
        Then applies basic preprocessing (expensive operations like author disambiguation done after filtering).
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
                log_print(f"Skipping malformed entry due to error: {e}", level="warning")

        self.df = pd.DataFrame(rows)