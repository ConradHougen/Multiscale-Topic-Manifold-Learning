"""
model_evaluation.py

Implements multiple algorithms and utilities related to assessments of topic coherence, document density, document topic
entropy, and other related metrics which could help assess the utility of different aspects of a particular topic
modeling or embedding method. One such aspect of consideration is the tokenization process and choice of a vocabulary
during preprocessing.

For example, filtering out stopwords from a document corpus has been shown in various studies to increase the
discernibility of topic vectors or word-frequency vectors, when using bag-of-words models like LDA. For information
retrieval, we would like to create a topic manifold which places similar documents near each other and far from
documents that are semantically-different.

Implicit in the assessment of "near" and "far" is the definition of an appropriate metric. Prior studies have used
distances such as cosine or Euclidean distance between document-topic or word-frequency vectors. However, it is
known in the statistics literature that these distances may be inappropriate for comparing pairs of vectors which
represent multinomial distributions (e.g. document-topic, author-topic, or word-frequency). Distances such as the
Hellinger distance, or other f-divergences, may give different results in terms of the density of embedding vectors.
"""
# ==============================================================================
# Imports
# ==============================================================================
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance


EPSILON = 1e-10

# ==============================================================================
# Primary Functions
# ==============================================================================
def document_space_density(doc_embeddings):
    """Uses the definition of document space density (DSD) pioneered in Wolfram and Zhang (2008), cited in "Vocabulary
    size and its effect on topic representation" (2017).

    DSD is the mean similarity of the input set of documents (represented as vectors) with their centroid.
    DSD = \frac{sum_i^n Sim(D_i, C)}{n}, where
        n is the total number of documents,
        D_i is the embedding vector associated with document i, in the high-dimensional space,
        C is the centroid of the document space, calculated as the arithmetic mean of the document vectors,
        and Sim(D_i, C) = 1/Dist(D_i, C), for Dist(D_i, C) != 0;
                        =              1, for Dist(D_i, C) == 0;
        and Dist(D_i, C) is the Manhattan distance, aka the L1 norm.

    :param doc_embeddings: numpy ndarray size (n, r), where r is the dimension of the embedding space.
    :return: (doc space density scalar, similarities vector)
    """
    return entity_embedding_density(doc_embeddings, similarity=dsd_similarity, centroid_method=arithmetic_mean)


def entity_embedding_density(entity_embeddings, similarity=None, centroid_method=None):
    """Inspired by document space density (DSD) from Wolfram (2008), cited in "Vocabulary size and its effect on topic
    representation" (2017).

    This implements similar ideas as DSD, but it also includes the flexibility to compute the similarity with any
    appropriate user-defined similarity function. Also, the centroid function can be something other than the simple
    arithmetic mean.

    If similarity is None, this will use the definition from DSD.
    If centroid_method is None, arithmetic mean will be used to compute the centroid.

    :param entity_embeddings: numpy ndarray of size (n, r), where r is the dimension of the embedding space.
    :param similarity: optional user-defined function from 2 r-dimensional vectors to a scalar value.
    :param centroid_method: optional user-defined function from n r-dimensional vectors to a scalar value.
    :return: (mean similarity, numpy array of similarities)
    """
    # Compute centroid
    if centroid_method:
        centroid = centroid_method(entity_embeddings).reshape((1, -1))
    else:
        print("Using default centroid computation based on arithmetic mean...")
        centroid = arithmetic_mean(entity_embeddings).reshape((1, -1))

    # Compute similarities
    if similarity:
        sims = np.apply_along_axis(lambda x: similarity(x, centroid), axis=1, arr=entity_embeddings)
    else:
        print("Using default similarity metric, dsd similarity...")
        sims = np.apply_along_axis(lambda x: dsd_similarity(x, centroid), axis=1, arr=entity_embeddings)

    # Compute mean similarity
    mean_sim = np.mean(sims)

    # Return mean similarity and row vector of similarities
    return mean_sim, sims


def single_vec_entropy(p):
    """Calculate entropy of a single probability vector."""
    # Add small epsilon to avoid log(0)
    p_safe = np.maximum(p, EPSILON)
    return -np.dot(p_safe, np.log(p_safe))


def mean_entropy(vecs):
    """Calculate mean entropy across multiple probability vectors."""
    entropies = np.apply_along_axis(single_vec_entropy, axis=1, arr=vecs)
    return np.mean(entropies)


def mean_cosine_similarity(vecs):
    """Calculate mean pairwise cosine similarity."""
    cos_sims = cosine_similarity(vecs)
    # Take the mean over all pairs of cosine similarities, excluding self similarities on diagonal
    mean_cos_sim = np.mean(cos_sims[~np.eye(cos_sims.shape[0], dtype=bool)])
    return mean_cos_sim


def pairwise_topic_similarity(topic_term_vecs):
    """This function is an alias for mean_cosine_similarity

    :param topic_term_vecs: (n, r) matrix of word frequency distributions on vocab of size r
    :return: mean of paired cosine similarities
    """
    return mean_cosine_similarity(topic_term_vecs)


def jensen_shannon_divergence(p, q):
    """Calculate Jensen-Shannon divergence between two probability distributions."""
    # Ensure inputs are numpy arrays and normalized
    p = np.array(p)
    q = np.array(q)
    
    # Normalize to ensure they sum to 1
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q
    
    # Compute M = (P + Q) / 2
    m = 0.5 * (p + q)
    
    # Add small epsilon to avoid log(0)
    p_safe = np.maximum(p, EPSILON)
    q_safe = np.maximum(q, EPSILON)
    m_safe = np.maximum(m, EPSILON)
    
    # Compute KL divergences
    kl_pm = np.sum(p_safe * np.log(p_safe / m_safe))
    kl_qm = np.sum(q_safe * np.log(q_safe / m_safe))
    
    return 0.5 * kl_pm + 0.5 * kl_qm


def topic_coherence_score(topic_words, texts, measure='c_v'):
    """
    Calculate topic coherence score using various measures.
    
    Args:
        topic_words: List of words representing the topic
        texts: List of tokenized documents
        measure: Coherence measure ('c_v', 'c_npmi', 'c_uci', 'u_mass')
        
    Returns:
        Coherence score
    """
    try:
        from gensim.models import CoherenceModel
        from gensim.corpora import Dictionary
        
        # Create dictionary and corpus
        dictionary = Dictionary(texts)
        
        # Compute coherence
        coherence_model = CoherenceModel(
            topics=[topic_words],
            texts=texts,
            dictionary=dictionary,
            coherence=measure
        )
        
        return coherence_model.get_coherence()
    
    except ImportError:
        print("Gensim not available for coherence computation")
        return 0.0


def perplexity_score(model, test_corpus):
    """
    Calculate perplexity score for a topic model.
    
    Args:
        model: Trained topic model (e.g., LDA)
        test_corpus: Test corpus in bag-of-words format
        
    Returns:
        Perplexity score
    """
    try:
        return model.log_perplexity(test_corpus)
    except AttributeError:
        print("Model does not support perplexity calculation")
        return float('inf')


def topic_diversity(topic_words_list, topk=10):
    """
    Calculate topic diversity - the percentage of unique words in top-k words of all topics.
    
    Args:
        topic_words_list: List of lists, each containing top words for a topic
        topk: Number of top words to consider per topic
        
    Returns:
        Topic diversity score (0-1)
    """
    if not topic_words_list:
        return 0.0
    
    # Get top-k words for each topic
    all_topk_words = set()
    total_topk_words = 0
    
    for topic_words in topic_words_list:
        topk_words = topic_words[:topk]
        all_topk_words.update(topk_words)
        total_topk_words += len(topk_words)
    
    if total_topk_words == 0:
        return 0.0
    
    return len(all_topk_words) / total_topk_words


def silhouette_score_topics(doc_topic_matrix, doc_labels=None):
    """
    Calculate silhouette score for topic clustering.
    
    Args:
        doc_topic_matrix: Document-topic probability matrix
        doc_labels: True document labels (if available)
        
    Returns:
        Silhouette score
    """
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans
        
        if doc_labels is None:
            # Use topic assignments as labels
            doc_labels = np.argmax(doc_topic_matrix, axis=1)
        
        # Calculate silhouette score
        if len(np.unique(doc_labels)) > 1:
            return silhouette_score(doc_topic_matrix, doc_labels)
        else:
            return 0.0
            
    except ImportError:
        print("Scikit-learn not available for silhouette score")
        return 0.0


# ==============================================================================
# Utility Functions
# ==============================================================================
def manhattan_l1_distance(u, v):
    """Calculate Manhattan (L1) distance between two vectors."""
    return np.linalg.norm(u - v, ord=1)


def dsd_similarity(u, v):
    """Calculate DSD similarity based on Manhattan distance."""
    d = manhattan_l1_distance(u, v)
    if abs(d) > EPSILON:
        dsd_sim = 1/d
    else:
        dsd_sim = 1
    return dsd_sim


def euclidean_l2_distance(u, v):
    """Calculate Euclidean (L2) distance between two vectors."""
    return np.linalg.norm(u - v, ord=2)


def arithmetic_mean(vectors):
    """Calculate arithmetic mean of vectors."""
    return np.mean(vectors, axis=0)


def hellinger_distance(p, q):
    """Calculate Hellinger distance between two probability distributions."""
    p = np.array(p)
    q = np.array(q)
    
    # Normalize to ensure they sum to 1
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q
    
    # Compute Hellinger distance
    sqrt_p = np.sqrt(p)
    sqrt_q = np.sqrt(q)
    return np.sqrt(0.5 * np.sum((sqrt_p - sqrt_q) ** 2))


def bhattacharyya_distance(p, q):
    """Calculate Bhattacharyya distance between two probability distributions."""
    p = np.array(p)
    q = np.array(q)
    
    # Normalize to ensure they sum to 1
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q
    
    # Add small epsilon to avoid sqrt(0)
    p_safe = np.maximum(p, EPSILON)
    q_safe = np.maximum(q, EPSILON)
    
    # Compute Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p_safe * q_safe))
    
    # Compute Bhattacharyya distance
    return -np.log(bc) if bc > 0 else float('inf')


def evaluate_topic_model(model, test_corpus, texts, topic_words_list, doc_labels=None):
    """
    Comprehensive evaluation of a topic model.
    
    Args:
        model: Trained topic model
        test_corpus: Test corpus for perplexity calculation
        texts: List of tokenized documents for coherence calculation
        topic_words_list: List of top words for each topic
        doc_labels: True document labels (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Perplexity
    metrics['perplexity'] = perplexity_score(model, test_corpus)
    
    # Topic coherence (average across topics)
    coherences = []
    for topic_words in topic_words_list:
        coherence = topic_coherence_score(topic_words, texts)
        coherences.append(coherence)
    metrics['avg_coherence'] = np.mean(coherences) if coherences else 0.0
    
    # Topic diversity
    metrics['topic_diversity'] = topic_diversity(topic_words_list)
    
    # Document-topic matrix for additional metrics
    try:
        doc_topic_matrix = np.array([model.get_document_topics(doc, minimum_probability=0) 
                                   for doc in test_corpus])
        
        # Silhouette score
        metrics['silhouette_score'] = silhouette_score_topics(doc_topic_matrix, doc_labels)
        
        # Mean entropy of document-topic distributions
        metrics['mean_doc_entropy'] = mean_entropy(doc_topic_matrix)
        
    except Exception as e:
        print(f"Could not calculate document-topic metrics: {e}")
        metrics['silhouette_score'] = 0.0
        metrics['mean_doc_entropy'] = 0.0
    
    return metrics