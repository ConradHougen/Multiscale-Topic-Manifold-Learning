"""
Geometry-Driven Longitudinal Topic Model (GDLTM) implementation.

Implements functions related to "A Geometry-Driven Longitudinal Topic Model", 
as appeared in Harvard Data Science Review (HDSR) 2021.

The primary purpose of GDLTM is to take a set of documents in a corpus, and produce
visualizations using PHATE low-dimensional embeddings of Hellinger distances between topics,
where the topics are extracted from the corpus, time slice by time slice.
"""

import copy
import random
import numpy as np
import gensim
import nltk

# Optional imports - framework works without these
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    mpl = None
    plt = None

try:
    import phate
    HAS_PHATE = True
except ImportError:
    HAS_PHATE = False

try:
    import pyLDAvis.gensim_models
    HAS_PYLDAVIS = True
except ImportError:
    HAS_PYLDAVIS = False
import pyLDAvis
import seaborn as sns
import warnings
import pickle
import pandas as pd
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.neighbors import kneighbors_graph
from .utils import get_data_int_dir

warnings.filterwarnings('ignore')
nltk.download('stopwords')
np.set_printoptions(suppress=True)


def convert_year_to_yearidx(year, tlist):
    """Convert year to year index in time list."""
    return tlist.index(year)


class GdltmParams:
    """Parameters for GDLTM model."""
    
    def __init__(self, dset, dsub, ntopics=4, knnk=10, gamma=0.75):
        self.dset = dset
        self.dsub = dsub
        self.ntopics = ntopics
        self.knnk = knnk
        self.gamma = gamma

    def print_params(self, save_dir=None):
        """Print and optionally save parameters to file."""
        print("/*--- GDLTM Params ---*/")
        print(f"dset: {self.dset}")
        print(f"dsub: {self.dsub}")
        print(f"ntopics: {self.ntopics}")
        print(f"knnk: {self.knnk}")
        print(f"gamma: {self.gamma}")
        print("/*--------------------*/")

        if save_dir:
            with open(os.path.join(save_dir, 'gdltm_params.txt'), 'w') as f:
                f.write("/*--- GDLTM Params ---*/\n")
                f.write(f"dset: {self.dset}\n")
                f.write(f"dsub: {self.dsub}\n")
                f.write(f"ntopics: {self.ntopics}\n")
                f.write(f"knnk: {self.knnk}\n")
                f.write(f"gamma: {self.gamma}\n")
                f.write("/*--------------------*/")


class Gdltm:
    """Geometry-Driven Longitudinal Topic Model implementation."""
    
    def __init__(self, gdltm_params):
        self.params = copy.deepcopy(gdltm_params)
        self.int_dir = get_data_int_dir(gdltm_params.dset, gdltm_params.dsub)

        # Load preprocessed data
        self._load_preprocessed_data()
        
        # Initialize data structures
        self._initialize_data_structures()

        # Data that is initialized at run time
        self.exp_dir = None
        self.train_row_inds = None
        self.train_df = None
        self.train_dt1 = None
        self.train_dt2 = None
        self.train_trange = None
        self.train_tlist = None
        self.corpus_per_t = None
        self.data_per_t = None
        self.lda_per_t = None
        self.topic_per_t = None
        self.hellinger_dist_matrix = None
        self.phate_op = None
        self.phate_data = None
        self.cluster_labels = None

    def _load_preprocessed_data(self):
        """Load preprocessed data from pickle files."""
        data_files = [
            'main_df.pkl',
            'data_words.pkl', 
            'id2word.pkl',
            'authorId_to_author.pkl',
            'author_to_authorId.pkl'
        ]
        
        for file_name in data_files:
            file_path = os.path.join(self.int_dir, file_name)
            with open(file_path, 'rb') as f:
                attr_name = file_name.replace('.pkl', '')
                setattr(self, attr_name, pickle.load(f))

    def _initialize_data_structures(self):
        """Initialize basic data structures."""
        self.nwords = len(self.id2word)
        self.main_df['date'] = pd.to_datetime(self.main_df['date'])
        self.main_df = self.main_df.sort_values(by='date')
        self.main_df = self.main_df.reset_index(drop=True)
        self.dt1 = self.main_df['date'].iloc[0]
        self.dt2 = self.main_df['date'].iloc[-1]

    def set_experiment_directory(self, exp_dir):
        """Set the experiment directory."""
        self.exp_dir = exp_dir
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)

    def prepare_training_data(self, train_row_inds=None):
        """Prepare training data for the specified time range."""
        if train_row_inds is None:
            self.train_row_inds = list(range(len(self.main_df)))
        else:
            self.train_row_inds = train_row_inds
            
        self.train_df = self.main_df.iloc[self.train_row_inds]
        self.train_dt1 = self.train_df['date'].iloc[0]
        self.train_dt2 = self.train_df['date'].iloc[-1]
        
        # Create time range and list
        self.train_trange = pd.date_range(
            start=self.train_dt1.replace(day=1),
            end=self.train_dt2.replace(day=1),
            freq='YS'
        )
        self.train_tlist = [dt.year for dt in self.train_trange]

    def build_corpus_per_timeslice(self):
        """Build corpus for each time slice."""
        self.corpus_per_t = {}
        self.data_per_t = {}
        
        for year in self.train_tlist:
            year_df = self.train_df[self.train_df['date'].dt.year == year]
            year_indices = year_df.index.tolist()
            
            # Get corresponding data_words for this year
            year_data_words = [self.data_words[i] for i in year_indices if i < len(self.data_words)]
            
            # Create corpus
            corpus = [self.id2word.doc2bow(doc) for doc in year_data_words]
            
            self.corpus_per_t[year] = corpus
            self.data_per_t[year] = year_data_words

    def train_lda_models(self):
        """Train LDA models for each time slice."""
        self.lda_per_t = {}
        self.topic_per_t = {}
        
        for year in self.train_tlist:
            if len(self.corpus_per_t[year]) > 0:
                # Train LDA model
                lda_model = gensim.models.LdaModel(
                    corpus=self.corpus_per_t[year],
                    id2word=self.id2word,
                    num_topics=self.params.ntopics,
                    random_state=42,
                    passes=10,
                    alpha='auto',
                    per_word_topics=True
                )
                
                self.lda_per_t[year] = lda_model
                
                # Extract topic distributions
                topics = []
                for topic_id in range(self.params.ntopics):
                    topic_dist = lda_model.show_topic(topic_id, topn=self.nwords)
                    topic_vector = np.zeros(self.nwords)
                    for word, prob in topic_dist:
                        if word in self.id2word.token2id:
                            word_id = self.id2word.token2id[word]
                            topic_vector[word_id] = prob
                    topics.append(topic_vector)
                
                self.topic_per_t[year] = np.array(topics)

    def compute_hellinger_distances(self):
        """Compute Hellinger distances between topics across time slices."""
        all_topics = []
        topic_metadata = []
        
        for year in self.train_tlist:
            if year in self.topic_per_t:
                for topic_idx, topic in enumerate(self.topic_per_t[year]):
                    all_topics.append(topic)
                    topic_metadata.append((year, topic_idx))
        
        if len(all_topics) == 0:
            return
            
        all_topics = np.array(all_topics)
        n_topics = len(all_topics)
        
        # Compute Hellinger distance matrix
        hellinger_matrix = np.zeros((n_topics, n_topics))
        
        for i in range(n_topics):
            for j in range(n_topics):
                # Hellinger distance formula
                sqrt_p = np.sqrt(all_topics[i])
                sqrt_q = np.sqrt(all_topics[j])
                hellinger_dist = np.sqrt(0.5 * np.sum((sqrt_p - sqrt_q) ** 2))
                hellinger_matrix[i, j] = hellinger_dist
        
        self.hellinger_dist_matrix = hellinger_matrix
        self.topic_metadata = topic_metadata

    def apply_phate_embedding(self, n_components=2):
        """Apply PHATE embedding to the Hellinger distance matrix."""
        if self.hellinger_dist_matrix is None:
            raise ValueError("Must compute Hellinger distances first")
            
        # Use distance matrix as input to PHATE
        self.phate_op = phate.PHATE(
            n_components=n_components,
            knn=self.params.knnk,
            gamma=self.params.gamma,
            random_state=42
        )
        
        self.phate_data = self.phate_op.fit_transform(self.hellinger_dist_matrix)

    def cluster_topics(self, n_clusters=None):
        """Perform hierarchical clustering on topics."""
        if self.phate_data is None:
            raise ValueError("Must apply PHATE embedding first")
            
        from sklearn.cluster import AgglomerativeClustering
        
        if n_clusters is None:
            n_clusters = min(self.params.ntopics, len(self.phate_data) // 2)
            
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        self.cluster_labels = clustering.fit_predict(self.phate_data)

    def visualize_topic_evolution(self, save_path=None):
        """Create visualization of topic evolution."""
        if self.phate_data is None or self.cluster_labels is None:
            raise ValueError("Must run PHATE embedding and clustering first")
            
        plt.figure(figsize=(12, 8))
        
        # Create color map for clusters
        unique_clusters = np.unique(self.cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        
        # Plot points colored by cluster
        for i, cluster in enumerate(unique_clusters):
            mask = self.cluster_labels == cluster
            plt.scatter(
                self.phate_data[mask, 0], 
                self.phate_data[mask, 1],
                c=[colors[i]], 
                label=f'Cluster {cluster}',
                alpha=0.7,
                s=60
            )
        
        # Add year labels
        for i, (year, topic_idx) in enumerate(self.topic_metadata):
            plt.annotate(
                f'{year}_{topic_idx}',
                (self.phate_data[i, 0], self.phate_data[i, 1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )
        
        plt.xlabel('PHATE 1')
        plt.ylabel('PHATE 2')
        plt.title('Topic Evolution Visualization (GDLTM)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def run_full_pipeline(self, train_row_inds=None, exp_dir=None):
        """Run the complete GDLTM pipeline."""
        if exp_dir:
            self.set_experiment_directory(exp_dir)
            
        print("Preparing training data...")
        self.prepare_training_data(train_row_inds)
        
        print("Building corpus per time slice...")
        self.build_corpus_per_timeslice()
        
        print("Training LDA models...")
        self.train_lda_models()
        
        print("Computing Hellinger distances...")
        self.compute_hellinger_distances()
        
        print("Applying PHATE embedding...")
        self.apply_phate_embedding()
        
        print("Clustering topics...")
        self.cluster_topics()
        
        print("GDLTM pipeline complete!")
        
        if self.exp_dir:
            save_path = os.path.join(self.exp_dir, 'topic_evolution.png')
            self.visualize_topic_evolution(save_path)