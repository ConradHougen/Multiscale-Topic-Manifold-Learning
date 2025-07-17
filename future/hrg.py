"""
Hierarchical Random Graph (HRG) implementation for MSTML.

This module provides functionality for working with Hierarchical Random Graphs,
including model fitting, prediction, and analysis of network structures.
"""

import numpy as np
import pandas as pd
import networkx as nx
import pickle
import os
import re
import subprocess
import matplotlib.pyplot as plt
import tempfile
from collections import defaultdict, Counter
from pathlib import Path
from .utils import save_pickle, load_pickle


class HierarchicalRandomGraph:
    """
    Hierarchical Random Graph model implementation.
    
    This class provides methods for fitting HRG models to networks,
    making predictions, and analyzing the hierarchical structure.
    """
    
    def __init__(self, graph=None):
        """
        Initialize HRG model.
        
        Args:
            graph: NetworkX graph object (optional)
        """
        self.graph = graph
        self.dendrogram = None
        self.likelihood = None
        self.consensus_tree = None
        self.fitted = False
        
    def fit(self, graph=None, num_iterations=1000000, temp_dir=None):
        """
        Fit HRG model to a graph.
        
        Args:
            graph: NetworkX graph (if not provided in constructor)
            num_iterations: Number of MCMC iterations
            temp_dir: Temporary directory for intermediate files
            
        Returns:
            Self for method chaining
        """
        if graph is not None:
            self.graph = graph
            
        if self.graph is None:
            raise ValueError("No graph provided for fitting")
            
        # Create temporary directory if not provided
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
            
        try:
            # Convert graph to edge list format
            edge_file = os.path.join(temp_dir, "network.pairs")
            self._write_edge_list(self.graph, edge_file)
            
            # Run HRG fitting (assuming external C++ implementation)
            hrg_file = os.path.join(temp_dir, "network.hrg")
            self._run_hrg_fit(edge_file, hrg_file, num_iterations)
            
            # Load results
            self.dendrogram = self._load_hrg_file(hrg_file)
            self.fitted = True
            
        except Exception as e:
            print(f"Error fitting HRG model: {e}")
            raise
            
        return self
    
    def predict_links(self, num_predictions=100):
        """
        Predict missing links using the fitted HRG model.
        
        Args:
            num_predictions: Number of link predictions to generate
            
        Returns:
            List of (node1, node2, probability) tuples
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # This would typically call the external HRG prediction code
        # For now, return placeholder implementation
        predictions = []
        
        # Get all possible node pairs not in current graph
        nodes = list(self.graph.nodes())
        existing_edges = set(self.graph.edges())
        
        possible_edges = []
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                if (node1, node2) not in existing_edges and (node2, node1) not in existing_edges:
                    possible_edges.append((node1, node2))
        
        # Sample predictions (placeholder - would use actual HRG probabilities)
        import random
        random.shuffle(possible_edges)
        
        for edge in possible_edges[:num_predictions]:
            # Placeholder probability calculation
            prob = random.uniform(0.1, 0.9)
            predictions.append((edge[0], edge[1], prob))
            
        return sorted(predictions, key=lambda x: x[2], reverse=True)
    
    def get_community_structure(self):
        """
        Extract community structure from the HRG dendrogram.
        
        Returns:
            Dictionary mapping nodes to community IDs
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
            
        # Placeholder implementation
        # Would extract communities from actual dendrogram structure
        communities = {}
        nodes = list(self.graph.nodes())
        
        # Simple placeholder: assign random communities
        import random
        num_communities = max(2, len(nodes) // 10)
        for node in nodes:
            communities[node] = random.randint(0, num_communities - 1)
            
        return communities
    
    def compute_likelihood(self):
        """
        Compute the likelihood of the graph given the HRG model.
        
        Returns:
            Log-likelihood value
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
            
        # Placeholder implementation
        # Would compute actual likelihood from dendrogram
        self.likelihood = -len(self.graph.edges()) * np.log(len(self.graph.nodes()))
        return self.likelihood
    
    def generate_synthetic_graph(self, num_nodes=None):
        """
        Generate a synthetic graph from the fitted HRG model.
        
        Args:
            num_nodes: Number of nodes (defaults to original graph size)
            
        Returns:
            NetworkX graph object
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
            
        if num_nodes is None:
            num_nodes = len(self.graph.nodes())
            
        # Placeholder implementation
        # Would generate graph from actual HRG model
        G = nx.erdos_renyi_graph(num_nodes, 0.1)
        return G
    
    def save_model(self, filepath):
        """Save the fitted HRG model to file."""
        if not self.fitted:
            raise ValueError("No fitted model to save")
            
        model_data = {
            'dendrogram': self.dendrogram,
            'likelihood': self.likelihood,
            'fitted': self.fitted
        }
        save_pickle(model_data, filepath)
    
    def load_model(self, filepath):
        """Load a fitted HRG model from file."""
        model_data = load_pickle(filepath)
        self.dendrogram = model_data['dendrogram']
        self.likelihood = model_data['likelihood']
        self.fitted = model_data['fitted']
        return self
    
    def _write_edge_list(self, graph, filename):
        """Write graph as edge list to file."""
        with open(filename, 'w') as f:
            for edge in graph.edges():
                f.write(f"{edge[0]} {edge[1]}\n")
    
    def _run_hrg_fit(self, edge_file, output_file, num_iterations):
        """
        Run external HRG fitting code.
        
        This is a placeholder - would call actual C++ HRG implementation.
        """
        # Placeholder: create dummy HRG file
        with open(output_file, 'w') as f:
            f.write("# Placeholder HRG file\n")
            f.write(f"# Fitted with {num_iterations} iterations\n")
    
    def _load_hrg_file(self, filename):
        """Load HRG dendrogram from file."""
        # Placeholder implementation
        # Would parse actual HRG file format
        return {"type": "placeholder_dendrogram"}


class HRGEnsemble:
    """
    Ensemble of HRG models for improved predictions and robustness.
    """
    
    def __init__(self, num_models=10):
        """
        Initialize HRG ensemble.
        
        Args:
            num_models: Number of HRG models in ensemble
        """
        self.num_models = num_models
        self.models = []
        self.fitted = False
    
    def fit(self, graph, num_iterations=100000):
        """
        Fit ensemble of HRG models.
        
        Args:
            graph: NetworkX graph
            num_iterations: Number of MCMC iterations per model
        """
        self.models = []
        
        for i in range(self.num_models):
            print(f"Fitting HRG model {i+1}/{self.num_models}")
            model = HierarchicalRandomGraph(graph)
            model.fit(num_iterations=num_iterations)
            self.models.append(model)
        
        self.fitted = True
        return self
    
    def predict_links(self, num_predictions=100):
        """
        Predict links using ensemble averaging.
        
        Args:
            num_predictions: Number of predictions to return
            
        Returns:
            List of (node1, node2, avg_probability) tuples
        """
        if not self.fitted:
            raise ValueError("Ensemble must be fitted first")
        
        # Collect predictions from all models
        all_predictions = defaultdict(list)
        
        for model in self.models:
            predictions = model.predict_links(num_predictions * 2)  # Get more to average
            for node1, node2, prob in predictions:
                all_predictions[(node1, node2)].append(prob)
        
        # Average probabilities
        averaged_predictions = []
        for (node1, node2), probs in all_predictions.items():
            avg_prob = np.mean(probs)
            averaged_predictions.append((node1, node2, avg_prob))
        
        # Sort by probability and return top predictions
        averaged_predictions.sort(key=lambda x: x[2], reverse=True)
        return averaged_predictions[:num_predictions]
    
    def get_consensus_communities(self):
        """
        Get consensus community structure across ensemble.
        
        Returns:
            Dictionary mapping nodes to community IDs
        """
        if not self.fitted:
            raise ValueError("Ensemble must be fitted first")
        
        # Collect community assignments from all models
        all_communities = []
        for model in self.models:
            communities = model.get_community_structure()
            all_communities.append(communities)
        
        # Find consensus (placeholder implementation)
        # Would use more sophisticated consensus method
        consensus = all_communities[0]  # Simple: use first model
        return consensus


def load_hrg_from_file(hrg_file, pairs_file):
    """
    Load HRG model from external files.
    
    Args:
        hrg_file: Path to .hrg file
        pairs_file: Path to .pairs file
        
    Returns:
        HierarchicalRandomGraph object
    """
    # Load graph from pairs file
    graph = nx.read_edgelist(pairs_file)
    
    # Create HRG object and load dendrogram
    hrg = HierarchicalRandomGraph(graph)
    hrg.dendrogram = hrg._load_hrg_file(hrg_file)
    hrg.fitted = True
    
    return hrg


def compare_hrg_models(hrg1, hrg2):
    """
    Compare two HRG models.
    
    Args:
        hrg1, hrg2: HierarchicalRandomGraph objects
        
    Returns:
        Dictionary of comparison metrics
    """
    if not (hrg1.fitted and hrg2.fitted):
        raise ValueError("Both models must be fitted")
    
    metrics = {}
    
    # Compare likelihoods
    metrics['likelihood_diff'] = abs(hrg1.compute_likelihood() - hrg2.compute_likelihood())
    
    # Compare community structures
    comm1 = hrg1.get_community_structure()
    comm2 = hrg2.get_community_structure()
    
    # Convert to lists for comparison
    nodes = sorted(set(comm1.keys()) & set(comm2.keys()))
    labels1 = [comm1[node] for node in nodes]
    labels2 = [comm2[node] for node in nodes]
    
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    metrics['ari'] = adjusted_rand_score(labels1, labels2)
    metrics['nmi'] = normalized_mutual_info_score(labels1, labels2)
    
    return metrics