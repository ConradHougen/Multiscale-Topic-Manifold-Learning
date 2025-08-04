# MSTML Package Documentation

This document provides a comprehensive overview of all classes, methods, and functions in the Multi-Scale Topic Manifold Learning (MSTML) package.

## Package Architecture

The MSTML package is organized into several core modules and specialized driver modules:

### Core Modules
- `core.py` - High-level workflow orchestration and user-friendly wrappers
- `author_disambiguation.py` - Author name disambiguation and ID assignment
- `dataframe_schema.py` - DataFrame schema definitions and extractors
- `data_loaders.py` - Data loading utilities for various formats
- `text_preprocessing.py` - Comprehensive text processing pipeline
- `model_evaluation.py` - Topic model evaluation metrics
- `utils.py` - General utility functions

### Driver Modules (Modular Function Libraries)
- `_math_driver.py` - Mathematical functions and distance metrics
- `_graph_driver.py` - Network analysis and graph operations
- `_embedding_driver.py` - Manifold learning and visualization
- `_topic_model_driver.py` - Topic modeling and HRG operations
- `_file_driver.py` - File I/O and directory management

---

## `author_disambiguation.py`

### Classes

#### `AuthorDisambiguator`
Fuzzy matching author name disambiguation system.

**Methods:**
- `__init__()` - Initialize disambiguator with configuration
- `_initialize_id_pool()` - Create shuffled ID pool
- `_expand_id_pool()` - Expand to next digit length
- `_get_next_id()` - Get unique ID with collision detection
- `_estimate_required_ids()` - Estimate needed unique IDs
- `_ensure_sufficient_pool_size()` - Ensure adequate pool capacity
- `update_dataframe()` - Main interface for data processing
- `fit()` - Train disambiguator on author lists
- `_extract_and_standardize_authors()` - Extract and clean author names
- `_group_authors_by_prefix()` - Group by first two characters
- `_compute_fuzzy_matches()` - TF-IDF cosine similarity matching
- `_assign_author_ids()` - Assign IDs based on matches
- `_get_author_ids_for_document()` - Get IDs for document authors
- `save_mappings()` - Save mappings to pickle files
- `load_mappings()` - Load mappings from pickle files
- `_sync_with_dataframe_ids()` - Internal dataframe synchronization method
- `sync_with_dataframe_ids()` - Public synchronization interface method
- `_ngrams()` - Generate character ngrams for matching
- `get_mapping_stats()` - Return mapping and pool statistics

---

## `dataframe_schema.py`

### Classes

#### `MainDataSchema` (Enum)
Defines pandas DataFrame column schemas.

**Fields:**
- `TITLE` - Document title extraction schema
- `DATE` - Document date extraction schema  
- `RAW_TEXT` - Raw text content schema
- `AUTHOR_NAMES` - Author names list schema
- `AUTHOR_IDS` - Author IDs list schema
- `PREPROCESSED_TEXT` - Processed text tokens schema

**Methods:**
- `colname` - Get column name property
- `get_extractor()` - Get field extraction function
- `all_colnames()` - Return all column names
- `all_fields()` - Return all schema fields

### Named Tuples

#### `FieldDef`
Schema field definition structure.

**Fields:**
- `column_name` - DataFrame column name
- `extractor` - Data extraction function
- `type` - Expected data type

---

## `data_loaders.py`

### Functions

#### Utility Functions
- `get_project_root_directory()` - Return project root path
- `get_data_directory()` - Return data folder path

### Classes

#### `DataLoader` (Abstract Base Class)
Base class for loading datasets.

**Methods:**
- `__init__()` - Initialize with input parameters
- `get_clean_df()` - Get validated rows DataFrame
- `get_na_df()` - Get invalid rows DataFrame
- `run()` - Execute complete data pipeline
- `dataset_root_dir` - Root directory property accessor
- `dataset_original_dir` - Original data directory property
- `dataset_clean_dir` - Clean data directory property
- `dataset_networks_dir` - Networks directory property accessor
- `setup_dataset_dirs()` - Create standard folder structure
- `_load_raw_data()` - Abstract raw data loading
- `_preprocess()` - Abstract data preprocessing method
- `_prepare_environment()` - Setup dataset folder structure
- `_prepare_input()` - Resolve and prepare inputs
- `_resolve_input_path()` - Handle input path resolution
- `_validate_and_flag()` - Validate data and flag
- `_save_outputs()` - Save processed data outputs
- `_generate_metadata()` - Create dataset summary metadata
- `_apply_preprocessors()` - Apply text and author processing

#### `JsonDataLoader` (DataLoader)
JSON file data loader implementation.

**Methods:**
- `_load_raw_data()` - Load JSON lines format
- `_preprocess()` - Convert entries to DataFrame

---

## `text_preprocessing.py`

### Functions

#### Utility Functions
- `super_simple_preprocess()` - Basic text tokenization function
- `lemmatize_doc()` - Lemmatize document tokens list
- `filter_doc_with_approved_tokens()` - Filter by approved vocabulary

### Classes

#### `TextPreprocessor`
Comprehensive text processing pipeline system.

**Methods:**
- `__init__()` - Initialize with stopwords configuration
- `filter_by_categories()` - Filter by document categories
- `preprocess_raw_text()` - Basic text tokenization step
- `lemmatize_all()` - Parallel document lemmatization process
- `build_dictionary()` - Create vocabulary dictionary mapping
- `compute_doc_frequencies()` - Calculate document frequency statistics
- `apply_frequency_stopword_filter()` - Remove high/low frequency terms
- `train_lda_model()` - Train LDA topic model
- `compute_relevancy_scores()` - Calculate term relevancy scores
- `apply_relevancy_filter()` - Filter by term relevance
- `drop_empty_rows()` - Remove empty text rows
- `get_dictionary()` - Return vocabulary dictionary object
- `get_corpus()` - Return bag-of-words corpus
- `get_vocab_size()` - Return vocabulary size count
- `get_stats_log()` - Return processing statistics log
- `get_processed_docs()` - Return filtered document tokens
- `get_lda_model()` - Return trained LDA model

---

## `core.py`

### Classes

#### `MSTMLConfig`
Configuration container for MSTML workflows.

**Methods:**
- `__init__()` - Initialize MSTML configuration with defaults

#### `TextProcessingWrapper`
User-friendly wrapper for text preprocessing functionality.

**Methods:**
- `__init__()` - Initialize text processing wrapper
- `process_dataframe()` - Process text with full preprocessing pipeline
- `get_vocabulary_stats()` - Get vocabulary statistics
- `get_dictionary()` - Get the vocabulary dictionary

#### `AuthorProcessingWrapper`
User-friendly wrapper for author disambiguation functionality.

**Methods:**
- `__init__()` - Initialize author processing wrapper
- `process_dataframe()` - Process author names to assign unique IDs
- `get_author_mappings()` - Get author name to ID mappings
- `get_mapping_stats()` - Get mapping statistics
- `save_mappings()` - Save author mappings to directory
- `load_mappings()` - Load author mappings from directory

#### `DataProcessingPipeline`
Integrated data processing pipeline combining all preprocessing steps.

**Methods:**
- `__init__()` - Initialize data processing pipeline
- `process_dataframe()` - Run complete data processing pipeline
- `get_processing_stats()` - Get comprehensive processing statistics
- `save_results()` - Save processing results to directory

#### `TemporalAnalysisWrapper`
User-friendly wrapper for temporal/longitudinal analysis.

**Methods:**
- `__init__()` - Initialize temporal analysis wrapper
- `create_temporal_windows()` - Split dataframe into temporal windows
- `analyze_vocabulary_evolution()` - Analyze vocabulary evolution over time

#### `NetworkAnalysisWrapper`
User-friendly wrapper for network analysis functionality.

**Methods:**
- `__init__()` - Initialize network analysis wrapper
- `build_coauthorship_network()` - Build coauthorship network from DataFrame
- `detect_communities()` - Detect communities in the network
- `analyze_interdisciplinarity()` - Analyze document interdisciplinarity

#### `MSTMLWorkflow`
High-level workflow orchestration for MSTML analysis.

**Methods:**
- `__init__()` - Initialize MSTML workflow
- `run_complete_analysis()` - Run complete MSTML analysis workflow
- `save_results()` - Save all analysis results

### Functions

**Convenience Functions:**
- `create_default_config()` - Create a default MSTML configuration
- `run_quick_analysis()` - Run MSTML analysis with default configuration

---

## `model_evaluation.py`

### Constants
- `EPSILON = 1e-10`

### Functions

**Density & Entropy Metrics:**
- `document_space_density()`
- `entity_embedding_density()`
- `single_vec_entropy()`
- `mean_entropy()`

**Similarity Measures:**
- `mean_cosine_similarity()`
- `pairwise_topic_similarity()`
- `jensen_shannon_divergence()`
- `manhattan_l1_distance()`
- `euclidean_l2_distance()`
- `hellinger_distance()`
- `bhattacharyya_distance()`
- `dsd_similarity()`

**Topic Model Evaluation:**
- `topic_coherence_score()`
- `perplexity_score()`
- `topic_diversity()`
- `silhouette_score_topics()`
- `evaluate_topic_model()`

**Utility Functions:**
- `arithmetic_mean()`

---

## `utils.py`

### Enums

#### `AuthEmbedEnum`
Author embedding representation types.

#### `TopicRelevanceEnum`
Topic representation filtering methods.

### Functions

**Dataset & I/O:**
- `validate_dataset_name()`
- `log_print()`
- `save_pickle()`
- `load_pickle()`

**Directory Management:**
- `get_data_int_dir()`
- `get_data_clean_dir()`
- `get_data_original_dir()`

**File Processing:**
- `get_file_path_without_extension()`
- `get_file_stem_only()`
- `create_hash_id()`

**Distance & Similarity:**
- `hellinger_distance()`
- `hellinger_similarity()`
- `jensen_shannon_divergence()`
- `cosine_similarity()`

**Text & Network Processing:**
- `preprocess_text()`
- `create_author_network()`
- `compute_network_metrics()`
- `normalize_vector()`

**Analysis Functions:**
- `create_time_windows()`
- `filter_by_frequency()`
- `compute_topic_coherence()`

---

## Driver Modules

### `_math_driver.py`

**Mathematical Functions and Distance Metrics (30+ functions)**

#### Distance Metrics & Similarity Functions
- `hellinger()` - Hellinger distance between distributions
- `hellinger_sim()` - Hellinger similarity measure
- `euclidean()` - Euclidean distance calculation
- `kl_divergence()` - Kullback-Leibler divergence
- `js_divergence()` - Jensen-Shannon divergence
- `min_weighted_tv()` - Minimum-weighted total variation
- `mean_weighted_tv()` - Mean-weighted total variation
- `atoms_term_relevance_stable()` - Stable term relevance computation

#### Statistical Measures
- `entropy()` - Shannon entropy of distributions
- `gini_coefficient()` - Gini coefficient calculation
- `max_weighted_mean()` - Max-weighted mean of vectors
- `entropy_of_max_weighted_mean()` - Entropy of max-weighted mean

#### Diffusion & Graph Processing
- `diffuse_distribution()` - Distribution diffusion over graphs
- `diffuse_distribution_parallel()` - Parallelized diffusion
- `precompute_weights()` - Graph weight precomputation
- `compute_new_distribution()` - Single node distribution update

#### Clustering Functions
- `hier_cluster()` - Hierarchical clustering
- `assign_clusters()` - Cluster assignment and labeling

#### Utility Functions
- `rescale_parameter()` - Parameter rescaling
- `intersection()` - Set intersection operations
- `pairwise_alignment_score()` - Topic alignment scoring
- `compute_alignment_scores()` - Multi-chunk alignment analysis
- `rerank_topic_terms_by_relevance()` - Term relevance reranking

---

### `_graph_driver.py`

**Network Analysis and Graph Operations (40+ functions)**

#### Network Construction & I/O
- `compose_coauthorship_network()` - Multi-year network composition
- `gen_nx_multigraphs_per_year()` - Year-based network generation
- `nx_graph_to_pairs_file()` - NetworkX to pairs file conversion
- `gen_nx_multigraph_from_dataframe()` - DataFrame to graph conversion
- `get_coauthors()` - Author collaboration retrieval

#### Hypergraph Functions
- `gen_hypergraph()` - Hypergraph generation from data
- `flatten_df_column_with_str_list()` - DataFrame flattening utility

#### Community Detection & Analysis
- `assign_topic_communities()` - Topic-based community assignment
- `assign_louvain_communities()` - Louvain method partitioning
- `compare_communities_adjusted_rand_index()` - ARI comparison
- `compare_communities_norm_mutual_information()` - NMI comparison
- `fixed_source_label_propagation()` - Constrained label propagation

#### Network Analysis & Metrics
- `find_kth_largest_connected_component()` - Component analysis
- `topic_space_dist_vs_path_length()` - Distance-path relationship
- `topic_space_dist_vs_path_len_for_non_overlapping_max_paths()` - Non-overlapping path analysis

#### Link Prediction & Evaluation
- `lp_expA_score()` - F1 score computation for link prediction

#### Graph File Format Conversion
- `graphml_viz_convert()` - Visualization format conversion
- `graphml_viz_convert_file()` - Single file conversion
- `convert_viz_graphml_to_pairs_file()` - GraphML to pairs conversion
- `atoms_read_graphml_to_netx()` - GraphML reader wrapper

#### Helper Functions
- `get_fname_for_single_yr_graph()` - Single year filename generation
- `get_fname_for_graph_composition()` - Composition filename generation
- `is_n_tuple()` - Tuple validation utility

---

### `_embedding_driver.py`

**Manifold Learning and Visualization (25+ functions)**

#### PHATE & Path Selection Functions
- `select_smooth_path()` - Smooth path selection in embeddings
- `refine_smooth_path()` - Path smoothing and refinement

#### Visualization Functions
- `generate_colors()` - Perceptually uniform color generation
- `plot_phate_embedding_with_filtered_chunks()` - Interactive PHATE plotting
- `plot_wordcloud_for_topic()` - Topic-specific word cloud generation

#### Word Cloud Functions
- `find_centroids_and_create_wordclouds()` - Cluster centroid word clouds
- `display_wordcloud()` - Word cloud display utility
- `create_and_save_wordcloud()` - Dataset word cloud creation
- `generate_slc_tpc_wordclouds()` - Slice-topic word clouds
- `generate_tpc_wordclouds()` - Topic word cloud generation

#### Utility Functions for Embeddings
- `convert_slc_tpc_idx_to_yr_and_tpc_idx()` - Index conversion utilities

---

### `_topic_model_driver.py`

**Topic Modeling and HRG Operations (50+ functions)**

#### Enums & Classes
- `AuthEmbedEnum` - Author embedding types
- `TermRelevanceTopicType` - Topic relevance filtering methods
- `TermRelevanceTopicFilter` - Topic filtering by term relevance

#### Document & Corpus Processing
- `preprocess_documents()` - Document preprocessing
- `expand_doc_topic_distns()` - Topic distribution expansion
- `map_chunk_and_topic_to_chunk_topic_index()` - Index mapping
- `count_unique_topics_or_categories()` - Topic counting utilities

#### Author Embedding Functions
- `author_word_freq_embedding()` - Word frequency embeddings
- `author_topic_distn()` - Author-topic distributions
- `author_topic_sparsified_distn_embedding()` - Sparsified embeddings
- `author_topics_by_term_relevance_embedding()` - Term relevance embeddings
- `author_vocab_by_term_relevance_embedding()` - Vocabulary-filtered embeddings

#### HRG & Dendrogram Functions
- `find_max_min_cut_distance()` - Cut distance analysis
- `truncate_dendrogram()` - Dendrogram truncation
- `get_leaf_nodes()` - Leaf node extraction
- `get_new_leaf_nodes()` - New leaf node identification
- `find_first_common_parent()` - Tree traversal utilities
- `setup_author_probs_matrix()` - Author probability matrix setup
- `setup_link_prob_matrix()` - Link probability matrix setup
- `compute_link_likelihood_scores()` - Link likelihood computation
- `save_dendrogram_and_index_map()` - Dendrogram serialization
- `load_dendrogram_and_index_map()` - Dendrogram deserialization

#### Interdisciplinarity & Analysis Functions
- `calculate_author_distributions()` - Author distribution calculation
- `score_interdisciplinarity()` - Interdisciplinarity scoring
- `calculate_major_n_topic_score()` - Major topic scoring

#### Utility Functions
- `get_nth_item_from_ordered_dict()` - OrderedDict utilities
- `get_doc_set_from_author_team()` - Document set retrieval
- `calculate_rank_correlations()` - Kendall's Tau correlations
- `get_top_percent()` - Top percentage extraction

---

### `_file_driver.py`

**File I/O and Directory Management (80+ functions)**

#### Directory & Path Management
- `get_net_src_dir()` - Network source directory
- `get_data_int_dir()` - Intermediate data directory
- `get_data_original_dir()` - Original data directory
- `get_data_intermediate_dir()` - Intermediate directory with subdirectories
- `get_data_networks_dir()` - Networks directory with subdirectories
- `get_exp_dir()` - Experiment directory generation
- `initialize_dataset_directories()` - Complete directory initialization
- `remove_empty_folders()` - Empty folder cleanup
- `create_exp_dir()` - Experiment directory creation

#### File I/O Operations
- `write_pickle()` - Pickle file writing with overwrite control
- `read_pickle()` - Pickle file reading
- `atoms_print()` - Dual console/file printing

#### Time & Naming Utilities
- `generate_unique_4digit_code()` - Time-based unique codes
- `get_date_hour_minute()` - Date-time string generation
- `get_hour_minute()` - Hour-minute time strings
- `get_hour_minute_second()` - Hour-minute-second strings
- `gen_time_str()` - Unique time strings
- `gen_date_str()` - Date-only strings
- `get_lda_model_file_name()` - LDA model naming

#### Path & File Utilities
- `get_file_path_without_extension()` - Extension-free paths
- `get_file_stem_only()` - Filename stem extraction
- `catalan_number()` - Catalan number computation

#### Parameter Management
- `init_inds()` - Parameter index initialization
- `incr_inds()` - Parameter index incrementation
- `gen_dir_suffix()` - Directory suffix generation
- `get_key()` - Dictionary key lookup

#### Data Processing Utilities
- `filter_dataframe()` - DataFrame filtering by constraints
- `load_filtered_dataframe()` - Filtered DataFrame loading
- `get_authorId2doc()` - Author ID to document mapping
- `get_author2doc()` - Author name to document mapping
- `merge_lists_to_set()` - List merging to sets

#### Text Processing Utilities
- `super_simple_preprocess()` - Simple text preprocessing
- `sent_to_words()` - Sentence tokenization
- `filter_list_bisect()` - Bisect-based filtering
- `remove_stopwords()` - Stopword removal
- `filter_stopwords_by_approval()` - Approved word filtering
- `remove_singly_occurring_words()` - Rare word removal
- `lemmatize_and_update()` - Lemmatization with vocabulary update
- `lemmatize_mp()` - Multiprocessing lemmatization
- `count_doc_freq_of_words()` - Document frequency counting
- `filter_terms()` - Vocabulary-based term filtering

#### PDF Operations
- `crop_pdf()` - PDF cropping with dimension validation

#### Data Filtering & Analysis Utilities
- `filter_doc_df_by_specified_authors()` - Author-based filtering
- `filter_all_by_connected_authors()` - Connected author filtering
- `filter_all_by_num_publications()` - Publication count filtering
- `filter_all_by_num_direct_coauthors()` - Coauthor degree filtering
- `filter_all_by_connected_component_size()` - Component size filtering
- `filter_all_by_connected_component_density()` - Component density filtering

#### Statistics & Analysis Functions (Placeholders)
- `stats_local_node_connectivity()` - Node connectivity statistics
- `stats_clustering_coefficients()` - Clustering coefficient statistics
- `stats_degree_assortativity()` - Degree assortativity statistics
- `stats_neighborhood_degree()` - Neighborhood degree statistics

---

## `__init__.py`

### Constants
- `__version__ = "1.0.0"`
- `__all__` (exported modules list)
- `_lazy_submodules` (lazy loading mapping)

### Functions
- `__getattr__()` - Lazy module loading
- `__dir__()` - Directory listing support

---

## `fast_encode_tree/__init__.py`

Import wrapper for fast tree encoding (Cython/Python fallback).

---

## `fast_encode_tree/fast_encode_tree_py.py`

### Classes

#### `TreeNode`
Tree structure node representation.

**Methods:**
- `__init__()`
- `__reduce__()`
- `__setstate__()`
- `is_leaf()`
- `get_leaf_count()`
- `get_all_leaf_ids()`

### Functions

**Tree Operations:**
- `calculate_left_right_link_prob()`
- `fast_encode_tree_structure()`
- `traverse_tree_preorder()`
- `traverse_tree_postorder()`
- `get_tree_depth()`
- `print_tree_structure()`

**I/O Functions:**
- `save_tree_structure()`
- `load_tree_structure()`