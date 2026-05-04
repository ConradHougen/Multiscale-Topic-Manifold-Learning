"""
legacy/comparison — Baseline comparison experiments for the MSTML legacy pipeline.

Compares MSTML (LDA + Hellinger-PHATE) against BERTopic on three dimensions:
  alignment   — forward k-NN topic matching across time chunks
  coherence   — gensim C_v, C_uci, C_npmi coherence scores
  diversity   — Diversity@k and mean pairwise Hellinger / cosine distance
"""
