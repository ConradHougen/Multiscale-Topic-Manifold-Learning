"""
Core MSTML (Multi-Scale Topic Manifold Learning) functionality.

This module contains the main classes and enums for the MSTML framework,
refactored from the original AToMS research code.
"""

import copy
import pickle
import os
from abc import ABC, abstractmethod
from .gdltm import Gdltm, GdltmParams
from .utils import get_data_int_dir
from enum import Enum


class MstmlEmbedType(Enum):
    """Enum for ways to create an embedding from pairs of document author teams."""
    WF_DISTN = 1
    GLOB_LDA_TPC_DISTN = 2
    SLC_TPC_HELLINGER_SIM_DISTN = 3
    SLC_TPC_HELLINGER_SIM_RAW = 4


class MstmlParams:
    """Input parameters for creating an MSTML object."""
    
    def __init__(self, gdltm_params, fwd_window=3, embed_type=MstmlEmbedType.WF_DISTN,
                 alpha=0.5, beta=1):
        self.dset = gdltm_params.dset
        self.dsub = gdltm_params.dsub
        self.gdltm_params = copy.deepcopy(gdltm_params)
        self.fwd_window = fwd_window
        self.embed_type = embed_type
        self.alpha = alpha
        self.beta = beta

    def print_params(self, save_dir=None):
        """Print and optionally save parameters to file."""
        print("/*---MSTML params---*/")
        self.gdltm_params.print_params()
        print(f"fwd_window: {self.fwd_window}")
        print(f"embed_type: {self.embed_type}")
        print(f"alpha: {self.alpha}")
        print(f"beta: {self.beta}")
        print("/*------------------*/")

        if save_dir:
            with open(os.path.join(save_dir, 'mstml_params.txt'), 'w') as f:
                f.write("/*---MSTML params---*/\n")
                f.write(f"fwd_window: {self.fwd_window}\n")
                f.write(f"embed_type: {self.embed_type}\n")
                f.write(f"alpha: {self.alpha}\n")
                f.write(f"beta: {self.beta}\n")
                f.write("/*------------------*/")


class MstmlPrinter:
    """Utility class for printing and saving experiment outputs."""
    
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.out_file = os.path.join(out_dir, "output.txt")

    def print(self, str_in, append=True):
        """Print to console and file."""
        print(str_in)
        self.f_print(str_in, append)

    def f_print(self, str_in, append=True):
        """Print to file only."""
        str_str_in = str(str_in)
        mode = 'a' if append and os.path.exists(self.out_file) else 'w'
        with open(self.out_file, mode) as f:
            f.write(str_str_in + '\n')

    def pkl_var(self, var_to_pkl, pkl_file_name):
        """Save variable to pickle file."""
        if not pkl_file_name.endswith(".pkl"):
            pkl_file_name = pkl_file_name + ".pkl"

        with open(os.path.join(self.out_dir, pkl_file_name), 'wb') as f:
            pickle.dump(var_to_pkl, f)


class Mstml(ABC):
    """Base class definition for MSTML experiments."""
    
    def __init__(self, params):
        self.params = copy.deepcopy(params)
        self.dset = params.dset
        self.dsub = params.dsub

        # Create data directories if needed
        self.int_dir = get_data_int_dir(params.dset, params.dsub)
        self.exp_dir = self.get_exp_dir()

        if not os.path.exists(self.int_dir):
            print(f"Creating intermediate data dir for MSTML object: {self.int_dir}")
            os.makedirs(self.int_dir, exist_ok=True)

        if not os.path.exists(self.exp_dir):
            print(f"Creating experiment dir for MSTML object: {self.exp_dir}")
            os.makedirs(self.exp_dir, exist_ok=True)

        self.printer = MstmlPrinter(self.exp_dir)

    @abstractmethod
    def get_exp_dir(self):
        """Get experiment directory path. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def run_experiment(self):
        """Run the experiment. Must be implemented by subclasses."""
        pass


class MstmlEnsembleInterdisciplinarity(Mstml):
    """MSTML experiment class for ensemble interdisciplinarity analysis."""
    
    def __init__(self, params, cut_height=0.5, n_hot=1, min_hot_threshold=0.2):
        super().__init__(params)
        self.cut_height = cut_height
        self.n_hot = n_hot
        self.min_hot_threshold = min_hot_threshold
        
    def get_exp_dir(self):
        """Get experiment directory for ensemble interdisciplinarity."""
        return os.path.join(
            self.int_dir, 
            f"ensemble_interdisciplinarity_cut{self.cut_height}_nhot{self.n_hot}"
        )
    
    def run_experiment(self):
        """Run ensemble interdisciplinarity experiment."""
        from .mstml_library import (
            score_interdisciplinarity, 
            compute_interdisciplinarity_score_fast,
            compute_pairwise_interdisciplinarity
        )
        
        self.printer.print("Starting Ensemble Interdisciplinarity Analysis")
        self.printer.print(f"Cut height: {self.cut_height}")
        self.printer.print(f"N-hot: {self.n_hot}")
        self.printer.print(f"Min hot threshold: {self.min_hot_threshold}")
        
        # Implementation would go here - this is a framework for the experiment
        # Users would need to provide their specific data and call the appropriate functions
        
        self.printer.print("Ensemble Interdisciplinarity Analysis completed")


class MstmlLongitudinalAnalysis(Mstml):
    """MSTML experiment class for longitudinal analysis."""
    
    def __init__(self, params, time_chunks=None, phate_params=None):
        super().__init__(params)
        self.time_chunks = time_chunks or []
        self.phate_params = phate_params or {'n_components': 2, 'knn': 5}
        
    def get_exp_dir(self):
        """Get experiment directory for longitudinal analysis."""
        return os.path.join(
            self.int_dir, 
            f"longitudinal_analysis_{len(self.time_chunks)}chunks"
        )
    
    def run_experiment(self):
        """Run longitudinal analysis experiment."""
        from .mstml_library import (
            get_chunk_to_meta_mapping,
            get_meta_topic_distributions,
            plot_phate_embedding_with_filtered_chunks,
            select_smooth_path
        )
        
        self.printer.print("Starting Longitudinal Analysis")
        self.printer.print(f"Number of time chunks: {len(self.time_chunks)}")
        self.printer.print(f"PHATE parameters: {self.phate_params}")
        
        # Implementation would go here - this is a framework for the experiment
        # Users would need to provide their specific data and call the appropriate functions
        
        self.printer.print("Longitudinal Analysis completed")