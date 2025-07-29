# MSTML Package Documentation

This document provides a comprehensive overview of all classes, methods, and functions in the Multi-Scale Topic Manifold Learning (MSTML) package.

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
- `standardize_authors()` - Format author name strings
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

## `gdltm_utils.py`

### Enums

#### `AuthEmbedEnum`
Author embedding representation types.

**Values:**
- `WORD_FREQ` - Word frequency embeddings
- `AT_DISTN` - Author-topic distribution embeddings
- `AT_SPARSIFIED_DISTN` - Sparsified author-topic distributions
- `TERM_RELEVANCE_N_HOT` - Term relevance n-hot encoding
- `TERM_RELEVANCE_VMASK` - Term relevance vocabulary masking

#### `TermRelevanceTopicType`
Topic representation filtering methods.

**Values:**
- `N_HOT_ENCODING` - N most relevant terms
- `VOCAB_MASK` - Union vocabulary masking approach

### Classes

#### `TermRelevanceTopicFilter`
Topic filtering by term relevance.

**Methods:**
- `__init__()`
- `get_topics()`
- `get_trtt()`

### Functions (120+ utility functions)

**Code Generation & IDs:**
- `generate_unique_4digit_code()`
- `generate_colors()`
- `catalan_number()`

**File & Path Utilities:**
- `get_file_path_without_extension()`
- `get_file_stem_only()`
- `get_net_src_dir()`
- `get_data_int_dir()`
- `get_exp_dir()`
- `create_exp_dir()`

**Author Embeddings:**
- `author_word_freq_embedding()`
- `author_topic_distn()`
- `author_topic_sparsified_distn_embedding()`
- `author_topics_by_term_relevance_embedding()`
- `author_vocab_by_term_relevance_embedding()`

**Data Processing:**
- `filter_dataframe()`
- `load_filtered_dataframe()`
- `merge_lists_to_set()`
- `super_simple_preprocess()`
- `sent_to_words()`
- `remove_stopwords()`
- `filter_stopwords_by_approval()`
- `lemmatize_and_update()`
- `lemmatize_mp()`

**Distance & Similarity Metrics:**
- `kl_divergence()`
- `js_divergence()`
- `euclidean()`
- `hellinger()`
- `hellinger_sim()`
- `entropy()`
- `gini_coefficient()`

**Network Analysis:**
- `compose_coauthorship_network()`
- `gen_nx_multigraph_from_dataframe()`
- `gen_nx_multigraphs_per_year()`
- `gen_hypergraph()`
- `get_coauthors()`
- `fixed_source_label_propagation()`

**Clustering & Analysis:**
- `hier_cluster()`
- `assign_clusters()`
- `find_kth_largest_connected_component()`

**I/O & Serialization:**
- `write_pickle()`
- `read_pickle()`
- `nx_graph_to_pairs_file()`
- `crop_pdf()`

**Visualization:**
- `generate_slc_tpc_wordclouds()`
- `generate_tpc_wordclouds()`
- `create_and_save_wordcloud()`
- `graphml_viz_convert()`

**Statistical Analysis:**
- `stats_local_node_connectivity()`
- `stats_clustering_coefficients()`
- `stats_degree_assortativity()`
- `stats_neighborhood_degree()`

**Filtering Functions:**
- `filter_all_by_connected_authors()`
- `filter_all_by_num_publications()`
- `filter_all_by_connected_component_size()`
- `filter_all_by_connected_component_density()`

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

## `mstml_utils.py`

### Functions (60+ specialized MSTML functions)

**Document Processing:**
- `preprocess_documents()`
- `expand_doc_topic_distns()`
- `mstml_term_relevance_stable()`
- `precompute_weights()`

**Distribution Operations:**
- `diffuse_distribution()`
- `compute_new_distribution()`
- `diffuse_distribution_parallel()`
- `normalize_vector()`

**Hierarchical Analysis:**
- `truncate_dendrogram()`
- `get_leaf_nodes()`
- `get_new_leaf_nodes()`
- `find_first_common_parent()`
- `save_dendrogram_and_index_map()`
- `load_dendrogram_and_index_map()`

**Author & Link Analysis:**
- `setup_author_probs_matrix()`
- `setup_link_prob_matrix()`
- `compute_link_likelihood_scores()`
- `calculate_author_distributions()`

**Interdisciplinarity Scoring:**
- `score_interdisciplinarity()`
- `calculate_major_n_topic_score()`
- `compute_interdisciplinarity_score_fast()`
- `compute_pairwise_interdisciplinarity()`

**Community Detection:**
- `assign_topic_communities()`
- `assign_louvain_communities()`
- `compare_communities_adjusted_rand_index()`
- `compare_communities_norm_mutual_information()`

**Visualization & Analysis:**
- `find_centroids_and_create_wordclouds()`
- `display_wordcloud()`
- `plot_phate_embedding_with_filtered_chunks()`
- `plot_wordcloud_for_topic()`

**Path & Smoothing Operations:**
- `refine_smooth_path()`
- `select_smooth_path()`
- `find_max_min_cut_distance()`
- `rescale_parameter()`

**Mapping & Indexing:**
- `map_chunk_and_topic_to_chunk_topic_index()`
- `get_chunk_to_meta_mapping()`
- `get_meta_topic_distributions()`

---

## `utils.py`

### Enums

#### `AuthEmbedEnum`
Author embedding representation types (duplicate of gdltm_utils version).

#### `TopicRelevanceEnum`
Topic representation filtering methods (similar to gdltm_utils version).

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