import numpy as np
import pandas as pd
import random
import re
from ._file_driver import log_print, write_pickle, read_pickle
import os
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, find
from sparse_dot_topn import awesome_cossim_topn

class AuthorDisambiguator:
    def __init__(self, 
                 max_authors_per_doc: int = 20, 
                 similarity_threshold: float = 0.90, 
                 name_max_length: int = 20,
                 ngram_size: int = 3,
                 initial_id_digits: int = 7):
        # Validate input parameters
        if max_authors_per_doc <= 0:
            raise ValueError("max_authors_per_doc must be positive")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if name_max_length <= 0:
            raise ValueError("name_max_length must be positive")
        if ngram_size <= 0:
            raise ValueError("ngram_size must be positive")
        if initial_id_digits < 4:
            raise ValueError("initial_id_digits must be at least 4")
            
        # Configuration parameters
        self.max_authors_per_doc = max_authors_per_doc
        self.similarity_threshold = similarity_threshold
        self.name_max_length = name_max_length
        self.ngram_size = ngram_size
        self.initial_id_digits = initial_id_digits
        
        # Dictionaries for mapping author names to unique IDs
        # One ID may be mapped to multiple author names
        self.author_name_to_id = {}
        self.author_id_to_names = {}
        
        # ID pool management
        self._current_id_digits = initial_id_digits
        self._id_pool = []  # Pre-shuffled pool of available IDs
        self._pool_index = 0  # Current position in the pool
        self._used_ids = set()  # Track all used IDs to prevent collisions
        self._initialize_id_pool()
        
        # Internal state for processing
        self._is_fitted = False

    def _initialize_id_pool(self) -> None:
        """
        Initialize a shuffled pool of author IDs for the current digit length.
        
        This creates all possible IDs for the current digit length and shuffles them
        to provide pseudo-random allocation without collision checking.
        """
        # Calculate range for current digit length
        min_id = 10 ** (self._current_id_digits - 1)  # e.g., 1000000 for 7 digits
        max_id = (10 ** self._current_id_digits) - 1   # e.g., 9999999 for 7 digits
        
        # Create pool of all possible IDs and shuffle
        self._id_pool = list(range(min_id, max_id + 1))
        random.shuffle(self._id_pool)
        self._pool_index = 0
        
        log_print(f"Initialized ID pool with {len(self._id_pool)} IDs ({self._current_id_digits} digits)", level="debug")
    
    def _expand_id_pool(self) -> None:
        """
        Expand to the next digit length when current pool is exhausted.
        
        This method increases the digit length by 1 and creates a new shuffled pool.
        All existing IDs remain valid, but new IDs will have the new length.
        """
        old_digits = self._current_id_digits
        self._current_id_digits += 1
        
        # Reinitialize pool with new digit length
        self._initialize_id_pool()
        
        log_print(f"Expanded ID pool from {old_digits} to {self._current_id_digits} digits", level="info")
    
    def _get_next_id(self) -> int:
        """
        Get the next available ID from the pre-shuffled pool with collision detection.
        
        This method provides O(1) performance in the common case but includes
        collision detection to handle cases where the disambiguator and dataframe
        may be out of sync.
        
        Returns:
            Next unique integer ID
            
        Raises:
            RuntimeError: If unable to allocate unique ID
        """
        # Keep trying until we find an unused ID or exhaust the pool
        while self._pool_index < len(self._id_pool):
            candidate_id = self._id_pool[self._pool_index]
            self._pool_index += 1
            
            # Check for collision with existing IDs
            if candidate_id not in self._used_ids:
                self._used_ids.add(candidate_id)
                return candidate_id
            else:
                # Collision detected - this ID is already in use
                log_print(f"ID collision detected for {candidate_id}, trying next", level="debug")
                continue
        
        # If we've exhausted the current pool, expand to next digit length
        log_print("Current ID pool exhausted, expanding to next digit length", level="info")
        self._expand_id_pool()
        
        # Try again with expanded pool
        if self._pool_index < len(self._id_pool):
            candidate_id = self._id_pool[self._pool_index]
            self._pool_index += 1
            if candidate_id not in self._used_ids:
                self._used_ids.add(candidate_id)
                return candidate_id
                
        raise RuntimeError("Unable to allocate unique ID - pool expansion failed")
    
    def _estimate_required_ids(self, author_lists: List[List[str]]) -> int:
        """
        Estimate the number of unique author IDs that will be needed.
        
        This is used to potentially expand the ID pool before processing
        to avoid multiple expansions during fitting.
        
        Args:
            author_lists: List of author name lists to process
            
        Returns:
            Estimated number of unique IDs needed
        """
        if not author_lists:
            return 0
            
        # Extract and standardize all authors to get rough unique count
        all_authors = self._extract_and_standardize_authors(author_lists)
        unique_authors = set(all_authors)
        
        # Estimate that disambiguation will reduce this by roughly 10-30%
        # (conservative estimate - actual reduction depends on similarity)
        estimated_unique_ids = int(len(unique_authors) * 0.8)
        
        return estimated_unique_ids
    
    def _ensure_sufficient_pool_size(self, estimated_ids_needed: int) -> None:
        """
        Ensure the ID pool is large enough for the estimated number of IDs needed.
        
        Args:
            estimated_ids_needed: Estimated number of unique IDs that will be needed
        """
        available_ids = len(self._id_pool) - self._pool_index
        
        while available_ids < estimated_ids_needed:
            self._expand_id_pool()
            available_ids = len(self._id_pool) - self._pool_index
            
        log_print(f"Pool has {available_ids} available IDs for estimated need of {estimated_ids_needed}", level="debug")

    def update_dataframe(self, df: pd.DataFrame, author_names_column: str) -> pd.Series:
        """
        Main interface method for data_loaders.py integration.
        Takes a dataframe and processes the author_names column to return author IDs.
        
        Args:
            df: DataFrame containing author names
            author_names_column: Name of column containing lists of author names
            
        Returns:
            pd.Series: Series of author ID lists corresponding to each row
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If processing fails
        """
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
        if author_names_column not in df.columns:
            raise ValueError(f"Column '{author_names_column}' not found in DataFrame")
        
        # Auto-sync with any existing author IDs in the dataframe to prevent collisions
        # This handles cases where the dataframe has been modified externally
        author_ids_column = author_names_column.replace('_names', '_ids').replace('author_names', 'author_ids')
        if author_ids_column in df.columns:
            self._sync_with_dataframe_ids(df, author_ids_column)
            
        try:
            # Extract all author names from the dataframe
            all_author_lists = df[author_names_column].tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to extract author lists from DataFrame: {e}")
        
        # Filter out documents with too many authors
        valid_indices = []
        filtered_author_lists = []
        
        for idx, authors in enumerate(all_author_lists):
            if len(authors) <= self.max_authors_per_doc:
                valid_indices.append(idx)
                filtered_author_lists.append(authors)
            else:
                log_print(f"Skipping document at index {idx} with {len(authors)} authors (max: {self.max_authors_per_doc})", level="warning")
        
        # Perform disambiguation on valid author lists
        if not self._is_fitted:
            self.fit(filtered_author_lists)
        
        # Generate author IDs for each document
        result_series = pd.Series([[] for _ in range(len(df))], index=df.index)
        
        for result_idx, orig_idx in enumerate(valid_indices):
            authors = all_author_lists[orig_idx]
            author_ids = self._get_author_ids_for_document(authors)
            result_series.iloc[orig_idx] = author_ids
            
        return result_series
    
    def fit(self, author_lists: List[List[str]]) -> 'AuthorDisambiguator':
        """
        Fits the disambiguator on a collection of author name lists.
        
        Args:
            author_lists: List of lists, where each sublist contains author names for a document
            
        Returns:
            self: Returns the fitted disambiguator
            
        Raises:
            ValueError: If author_lists is invalid
            RuntimeError: If fitting process fails
        """
        if not author_lists:
            raise ValueError("author_lists cannot be empty")
        if not isinstance(author_lists, list):
            raise ValueError("author_lists must be a list")
        if not all(isinstance(sublist, list) for sublist in author_lists):
            raise ValueError("All elements in author_lists must be lists")
            
        try:
            # Extract and standardize all unique author names
            all_authors = self._extract_and_standardize_authors(author_lists)
            if not all_authors:
                raise ValueError("No valid authors found after standardization")
                
            unique_authors = sorted(list(set(all_authors)))
        except Exception as e:
            raise RuntimeError(f"Failed during author extraction and standardization: {e}")
        
        log_print(f"Number of authors prior to disambiguation: {len(unique_authors)}", level="info")
        
        # Estimate required IDs and ensure sufficient pool size
        estimated_ids = self._estimate_required_ids(author_lists)
        self._ensure_sufficient_pool_size(estimated_ids)
        
        # Group authors by first 2 characters for efficient processing
        authors_split = self._group_authors_by_prefix(unique_authors)
        
        # Perform fuzzy matching within each group
        fmatch_results = self._compute_fuzzy_matches(authors_split)
        
        # Assign unique IDs based on matches
        self.author_name_to_id, self.author_id_to_names = self._assign_author_ids(authors_split, fmatch_results)
        
        log_print(f"Number of authors after disambiguation: {len(self.author_id_to_names)}", level="info")
        
        self._is_fitted = True
        return self
    
    def _extract_and_standardize_authors(self, author_lists: List[List[str]]) -> List[str]:
        """
        Extract all author names from lists and standardize them.
        
        Args:
            author_lists: List of lists containing author names
            
        Returns:
            List of standardized author names
            
        Raises:
            ValueError: If input data is invalid
        """
        if not author_lists:
            raise ValueError("author_lists cannot be empty")
            
        all_authors = []
        
        for i, authors in enumerate(author_lists):
            if not isinstance(authors, list):
                log_print(f"Skipping non-list entry at index {i}: {type(authors)}", level="warning")
                continue
                
            for author in authors:
                if not isinstance(author, str):
                    log_print(f"Skipping non-string author at index {i}: {type(author)}", level="warning")
                    continue
                    
                author = author.strip()
                if not author:  # Skip empty strings
                    continue
                    
                # Only add authors whose names start with a letter
                if re.search(r"^[a-zA-Z]", author):
                    # Standardize: uppercase and truncate
                    standardized = author.upper()[:self.name_max_length].strip()
                    if standardized:  # Make sure we still have content after processing
                        all_authors.append(standardized)
                    
        return all_authors
    
    def _group_authors_by_prefix(self, unique_authors: List[str]) -> List[List[str]]:
        """
        Group authors by their first 2 characters for efficient processing.
        
        Args:
            unique_authors: List of unique standardized author names
            
        Returns:
            List of lists, each containing authors with the same 2-character prefix
            
        Raises:
            ValueError: If unique_authors is invalid
        """
        if not unique_authors:
            raise ValueError("unique_authors cannot be empty")
        if not all(isinstance(author, str) and len(author) >= 2 for author in unique_authors):
            raise ValueError("All authors must be strings with at least 2 characters")
            
        authors_split = []
        for prefix, group in groupby(unique_authors, key=lambda x: x[:2]):
            group_list = list(group)
            if group_list:  # Only add non-empty groups
                authors_split.append(group_list)
            
        log_print(f"Number of unique prefixes: {len(authors_split)}", level="info")
        return authors_split
    
    def _compute_fuzzy_matches(self, authors_split: List[List[str]]) -> List[Tuple]:
        """
        Compute fuzzy matches using TF-IDF vectorization and cosine similarity.
        
        Args:
            authors_split: List of author groups split by prefix
            
        Returns:
            List of tuples containing match results from scipy.sparse.find()
            
        Raises:
            ValueError: If authors_split is invalid
            RuntimeError: If vectorization or similarity computation fails
        """
        if not authors_split:
            raise ValueError("authors_split cannot be empty")
            
        fmatch_results = []
        
        for split_idx, author_group in enumerate(authors_split):
            if not author_group:
                log_print(f"Empty author group at index {split_idx}, skipping", level="warning")
                continue
                
            try:
                # Create TF-IDF matrix using character n-grams
                vectorizer = TfidfVectorizer(
                    min_df=1, 
                    analyzer=lambda x: self._ngrams(x, self.ngram_size)
                )
                tf_idf_matrix = vectorizer.fit_transform(author_group)
                
                if tf_idf_matrix.shape[0] == 0:
                    log_print(f"Empty TF-IDF matrix for group {split_idx}, skipping", level="warning")
                    continue
                
                # Compute similarity matrix
                topN = len(author_group)
                matches = awesome_cossim_topn(
                    tf_idf_matrix, 
                    tf_idf_matrix.transpose(), 
                    topN, 
                    self.similarity_threshold
                )
                
                log_print(f"Group {split_idx}: matrix shape {tf_idf_matrix.shape}, matches {matches.nnz}", level="debug")
                
                # Convert to find() format
                match_result = find(matches)
                fmatch_results.append(match_result)
                
            except Exception as e:
                log_print(f"Failed to compute matches for group {split_idx}: {e}", level="error")
                raise RuntimeError(f"Similarity computation failed for group {split_idx}: {e}")
            
        return fmatch_results
    
    def _assign_author_ids(self, authors_split: List[List[str]], fmatch_results: List[Tuple]) -> Tuple[Dict, Dict]:
        """
        Assign unique integer IDs to author names based on fuzzy matching results.
        
        Args:
            authors_split: List of author groups split by prefix
            fmatch_results: Results from fuzzy matching computation
            
        Returns:
            Tuple of (author_name_to_id, author_id_to_names) dictionaries
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If ID assignment fails
        """
        if len(authors_split) != len(fmatch_results):
            raise ValueError("authors_split and fmatch_results must have the same length")
            
        author_name_to_id = {}
        author_id_to_names = {}
        
        try:
            for author_group, fmatch in zip(authors_split, fmatch_results):
                if not author_group:
                    continue
                    
                # Validate fmatch structure
                if not isinstance(fmatch, tuple) or len(fmatch) != 3:
                    log_print(f"Invalid fmatch structure, skipping group", level="warning")
                    continue
                    
                for auth_idx, auth_name in enumerate(author_group):
                    if auth_name in author_name_to_id:
                        continue
                        
                    try:
                        # Find all authors that match this one
                        matching_indices = fmatch[1][fmatch[0] == auth_idx]
                        matching_names = [author_group[i] for i in matching_indices if i < len(author_group)]
                        
                        if not matching_names:
                            continue
                            
                        # Get next ID from pre-shuffled pool
                        auth_id = self._get_next_id()
                        
                        for name in matching_names:
                            if name not in author_name_to_id:
                                author_name_to_id[name] = auth_id
                                
                                if auth_id not in author_id_to_names:
                                    author_id_to_names[auth_id] = []
                                author_id_to_names[auth_id].append(name)
                                
                    except (IndexError, TypeError) as e:
                        log_print(f"Error processing author '{auth_name}' at index {auth_idx}: {e}", level="warning")
                        continue
                        
        except Exception as e:
            raise RuntimeError(f"Failed during author ID assignment: {e}")
                
        return author_name_to_id, author_id_to_names
    
    def _get_author_ids_for_document(self, authors: List[str]) -> List[int]:
        """
        Get author IDs for a single document's author list.
        
        Args:
            authors: List of author names for a document
            
        Returns:
            List of author IDs
            
        Raises:
            ValueError: If authors list is invalid
            RuntimeError: If disambiguator is not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Disambiguator must be fitted before getting author IDs")
        if not isinstance(authors, list):
            raise ValueError("authors must be a list")
            
        author_ids = []
        
        for author in authors:
            if not isinstance(author, str):
                log_print(f"Skipping non-string author: {type(author)}", level="warning")
                continue
                
            author = author.strip()
            if not author:
                continue
                
            # Standardize the author name
            standardized = author.upper()[:self.name_max_length].strip()
            
            # Skip if name doesn't start with letter
            if not re.search(r"^[a-zA-Z]", standardized):
                continue
                
            # Look up author ID
            if standardized in self.author_name_to_id:
                author_ids.append(self.author_name_to_id[standardized])
            else:
                log_print(f"Author '{standardized}' not found in mapping", level="warning")
                
        return author_ids
    
    def save_mappings(self, directory: str) -> None:
        """
        Save author mapping dictionaries and ID pool state to pickle files.
        
        Args:
            directory: Directory path to save mapping files
            
        Raises:
            ValueError: If directory path is invalid
            RuntimeError: If disambiguator is not fitted or save operation fails
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save mappings: disambiguator is not fitted")
        if not directory or not isinstance(directory, str):
            raise ValueError("directory must be a non-empty string")
            
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save mapping dictionaries
            name_to_id_path = os.path.join(directory, 'author_name_to_id.pkl')
            id_to_names_path = os.path.join(directory, 'author_id_to_names.pkl')
            
            write_pickle(self.author_name_to_id, name_to_id_path)
                
            write_pickle(self.author_id_to_names, id_to_names_path)
            
            # Save ID pool state
            pool_state_path = os.path.join(directory, 'id_pool_state.pkl')
            pool_state = {
                'current_id_digits': self._current_id_digits,
                'id_pool': self._id_pool,
                'pool_index': self._pool_index,
                'initial_id_digits': self.initial_id_digits
            }
            
            write_pickle(pool_state, pool_state_path)
                
            log_print(f"Saved author mappings and ID pool state to {directory}", level="info")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save mappings to {directory}: {e}")
    
    def load_mappings(self, directory: str) -> None:
        """
        Load author mapping dictionaries and ID pool state from pickle files.
        
        Args:
            directory: Directory path containing mapping files
            
        Raises:
            ValueError: If directory path is invalid
            FileNotFoundError: If mapping files don't exist
            RuntimeError: If loading fails
        """
        if not directory or not isinstance(directory, str):
            raise ValueError("directory must be a non-empty string")
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory does not exist: {directory}")
            
        name_to_id_path = os.path.join(directory, 'author_name_to_id.pkl')
        id_to_names_path = os.path.join(directory, 'author_id_to_names.pkl')
        pool_state_path = os.path.join(directory, 'id_pool_state.pkl')
        
        if not os.path.exists(name_to_id_path):
            raise FileNotFoundError(f"File not found: {name_to_id_path}")
        if not os.path.exists(id_to_names_path):
            raise FileNotFoundError(f"File not found: {id_to_names_path}")
            
        try:
            # Load mapping dictionaries
            loaded_name_to_id = read_pickle(name_to_id_path)
                
            loaded_id_to_names = read_pickle(id_to_names_path)
                
            # Validate loaded mapping data
            if not isinstance(loaded_name_to_id, dict):
                raise RuntimeError("Invalid author_name_to_id format")
            if not isinstance(loaded_id_to_names, dict):
                raise RuntimeError("Invalid author_id_to_names format")
                
            self.author_name_to_id = loaded_name_to_id
            self.author_id_to_names = loaded_id_to_names
            
            # Rebuild used IDs set from loaded mappings
            self._used_ids = set(self.author_id_to_names.keys())
            
            # Load ID pool state if available
            if os.path.exists(pool_state_path):
                pool_state = read_pickle(pool_state_path)
                    
                # Validate and restore pool state
                if isinstance(pool_state, dict):
                    self._current_id_digits = pool_state.get('current_id_digits', self.initial_id_digits)
                    self._id_pool = pool_state.get('id_pool', [])
                    self._pool_index = pool_state.get('pool_index', 0)
                    
                    # Update initial_id_digits if it was saved
                    if 'initial_id_digits' in pool_state:
                        self.initial_id_digits = pool_state['initial_id_digits']
                        
                    log_print(f"Restored ID pool state: {self._current_id_digits} digits, index {self._pool_index}", level="info")
                else:
                    log_print("Invalid pool state format, reinitializing pool", level="warning")
                    self._initialize_id_pool()
            else:
                log_print("No pool state file found, reinitializing pool", level="warning")
                self._initialize_id_pool()
            
            self._is_fitted = True
            log_print(f"Loaded author mappings from {directory}", level="info")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load mappings from {directory}: {e}")
    
    def _sync_with_dataframe_ids(self, df: pd.DataFrame, author_ids_column: str) -> None:
        """
        Internal method to synchronize the disambiguator with existing author IDs in a dataframe.
        
        This method performs bidirectional synchronization:
        1. Prevents collisions by tracking existing IDs
        2. Updates mappings with any new ID->name relationships found in dataframe
        3. Removes mappings for IDs no longer present in dataframe
        
        Args:
            df: DataFrame that may contain existing author IDs
            author_ids_column: Name of column containing author ID lists
        """
        if df is None or df.empty:
            return  # Nothing to sync
        if author_ids_column not in df.columns:
            log_print(f"Column '{author_ids_column}' not found, skipping sync", level="debug")
            return
        
        # Corresponding author names column
        author_names_column = author_ids_column.replace('_ids', '_names').replace('author_ids', 'author_names')
        if author_names_column not in df.columns:
            log_print(f"Corresponding names column '{author_names_column}' not found, partial sync only", level="debug")
            author_names_column = None
            
        try:
            # Extract all existing author IDs and their corresponding names from the dataframe
            dataframe_ids = set()
            dataframe_id_to_names = {}
            
            for idx, row in df.iterrows():
                author_ids = row[author_ids_column]
                author_names = row[author_names_column] if author_names_column else None
                
                if isinstance(author_ids, list):
                    names_list = author_names if isinstance(author_names, list) else []
                    
                    # Pair up IDs with names (handle mismatched lengths gracefully)
                    for i, aid in enumerate(author_ids):
                        if isinstance(aid, int):
                            dataframe_ids.add(aid)
                            
                            # If we have corresponding names, track the ID->name mapping
                            if i < len(names_list) and isinstance(names_list[i], str):
                                name = names_list[i].upper()[:self.name_max_length].strip()
                                if name and re.search(r"^[a-zA-Z]", name):
                                    if aid not in dataframe_id_to_names:
                                        dataframe_id_to_names[aid] = set()
                                    dataframe_id_to_names[aid].add(name)
            
            # 1. Update used IDs set for collision prevention
            old_used_count = len(self._used_ids)
            self._used_ids.update(dataframe_ids)
            new_used_count = len(self._used_ids)
            
            # 2. Update mappings with new ID->name relationships found in dataframe
            new_mappings_count = 0
            for aid, names_set in dataframe_id_to_names.items():
                if aid not in self.author_id_to_names:
                    # New ID found in dataframe - add to mappings
                    self.author_id_to_names[aid] = list(names_set)
                    for name in names_set:
                        self.author_name_to_id[name] = aid
                    new_mappings_count += 1
                    log_print(f"Added new mapping from dataframe: ID {aid} -> {list(names_set)}", level="debug")
                else:
                    # Existing ID - merge names
                    existing_names = set(self.author_id_to_names[aid])
                    new_names = names_set - existing_names
                    if new_names:
                        self.author_id_to_names[aid].extend(list(new_names))
                        for name in new_names:
                            self.author_name_to_id[name] = aid
                        log_print(f"Added names to existing ID {aid}: {list(new_names)}", level="debug")
            
            # 3. Remove mappings for IDs no longer present in dataframe
            # Only do this if we have a complete picture (both IDs and names columns exist)
            removed_ids = set()
            removed_names = set()
            
            if author_names_column and self.author_id_to_names:
                # IDs in our mappings but not in the current dataframe
                missing_ids = set(self.author_id_to_names.keys()) - dataframe_ids
                
                for aid in missing_ids:
                    # Remove this ID and all its associated names
                    if aid in self.author_id_to_names:
                        names_for_id = self.author_id_to_names[aid]
                        removed_ids.add(aid)
                        removed_names.update(names_for_id)
                        
                        # Remove from both mappings
                        del self.author_id_to_names[aid]
                        for name in names_for_id:
                            if name in self.author_name_to_id and self.author_name_to_id[name] == aid:
                                del self.author_name_to_id[name]
                        
                        # Remove from used IDs set
                        self._used_ids.discard(aid)
                
                if removed_ids:
                    log_print(f"Removed {len(removed_ids)} IDs no longer in dataframe: {sorted(removed_ids)}", level="info")
                    log_print(f"Removed names: {sorted(removed_names)}", level="debug")
            
            # Log summary
            if new_used_count > old_used_count or new_mappings_count > 0 or removed_ids:
                log_print(f"Dataframe sync complete: "
                           f"+{new_used_count - old_used_count} used IDs, "
                           f"+{new_mappings_count} new mappings, "
                           f"-{len(removed_ids)} removed mappings", level="info")
            else:
                log_print("Dataframe sync: no changes needed", level="debug")
                
        except Exception as e:
            log_print(f"Failed to sync with dataframe IDs: {e}", level="warning")
    
    def sync_with_dataframe_ids(self, df: pd.DataFrame, author_ids_column: str) -> None:
        """
        Public method to explicitly synchronize with existing author IDs in a dataframe.
        
        This is typically not needed as update_dataframe() automatically syncs,
        but can be useful for explicit control or verification.
        
        Args:
            df: DataFrame containing existing author IDs
            author_ids_column: Name of column containing author ID lists
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if not isinstance(author_ids_column, str):
            raise ValueError("author_ids_column must be a string")
            
        self._sync_with_dataframe_ids(df, author_ids_column)
    
    @staticmethod
    def _ngrams(string: str, n: int = 3) -> List[str]:
        """
        Generate character n-grams for fuzzy string matching.
        
        Args:
            string: Input string to generate n-grams from
            n: Size of n-grams to generate
            
        Returns:
            List of n-gram strings
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(string, str):
            raise ValueError("string must be a string")
        if n <= 0:
            raise ValueError("n must be positive")
        if len(string) < n:
            return []  # Can't generate n-grams if string is shorter than n
            
        # Clean and normalize the string
        string = string.encode("ascii", errors="ignore").decode()
        string = string.lower()
        chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
        rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
        string = re.sub(rx, '', string)
        string = string.replace('&', 'and')
        string = string.replace(',', ' ')
        string = string.replace('-', ' ')
        string = string.title()
        string = re.sub(' +', ' ', string).strip()
        string = ' ' + string + ' '
        string = re.sub(r'[,-./]|\sBD', r'', string)
        
        if len(string) < n:
            return []  # Return empty list if processed string is too short
            
        try:
            ngrams_list = zip(*[string[i:] for i in range(n)])
            return [''.join(ngram) for ngram in ngrams_list]
        except Exception as e:
            log_print(f"Error generating n-grams for string '{string}': {e}", level="warning")
            return []

    def get_mapping_stats(self) -> Dict[str, int]:
        """
        Get statistics about the current author mappings and ID pool.
        
        Returns:
            Dictionary containing mapping and pool statistics
            
        Raises:
            RuntimeError: If disambiguator is not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Disambiguator must be fitted to get stats")
            
        pool_utilization = (self._pool_index / len(self._id_pool)) * 100 if self._id_pool else 0
        
        return {
            'total_unique_authors': len(self.author_name_to_id),
            'total_author_ids': len(self.author_id_to_names),
            'avg_names_per_id': (len(self.author_name_to_id) / len(self.author_id_to_names) 
                               if self.author_id_to_names else 0),
            'current_id_digits': self._current_id_digits,
            'pool_size': len(self._id_pool),
            'pool_used': self._pool_index,
            'pool_remaining': len(self._id_pool) - self._pool_index,
            'pool_utilization_percent': round(pool_utilization, 2),
            'total_used_ids': len(self._used_ids),
            'collision_protection': True
        }