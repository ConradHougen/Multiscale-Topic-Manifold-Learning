# MSTML Legacy Pipeline

Self-contained reproduction of the CAMSAP 2025 paper results and dissertation
experiments. Fully independent of the `mstml/` library — no shared imports.

---

## Contents

```
legacy/
├── README.md                       ← this file
├── config.yaml                     ← stable secondary hyperparameter defaults
├── run_pipeline.py                 ← CLI entrypoint (6 resumable stages)
├── camsap2025_relational_topic_alignment.ipynb  ← original analysis notebook
├── _config.py                      ← PrimaryConfig + PipelineConfig dataclasses
├── _math.py                        ← Hellinger, entropy, FAISS bug fix, term relevance
├── _preprocessing.py               ← Stage 1: arXiv JSONL → processed DataFrame
├── _topic_models.py                ← Stage 2: LDA/BERTopic ensemble + expand distributions
├── _distributions.py               ← Stage 3: author barycenters, diffusion k-NN graph
├── _manifold.py                    ← Stage 4: dendrogram, PHATE/UMAP/t-SNE/PCA embedding
├── _scoring.py                     ← Stage 5: HRG interdisciplinarity scoring
├── _tree.py                        ← TreeNode + fast_encode_tree_structure (HRG)
├── _visualization.py               ← Static matplotlib/wordcloud figures
├── _interactive.py                 ← mplcursors + Plotly interactive plots
├── comparison/
│   ├── run_comparison.py           ← Comparison CLI (MSTML vs BERTopic)
│   ├── _bertopic_embed.py          ← BERTopic ensemble training
│   ├── _alignment.py               ← Forward k-NN alignment across chunks
│   ├── _coherence.py               ← Gensim C_v / C_npmi / C_uci coherence
│   └── _diversity.py               ← Diversity@k + mean pairwise distances
└── tests/
    ├── conftest.py                 ← Shared synthetic-data fixtures
    ├── pytest.ini                  ← Legacy-only pytest configuration
    ├── test_math.py                ← Unit tests: _math.py (42 tests)
    ├── test_pipeline.py            ← Integration tests: pipeline stages 1–5 (87 tests)
    ├── test_comparison.py          ← Comparison tests: MSTML vs BERTopic (59 tests)
    └── test_integration.py         ← MstmlOrchestrator regression tests (13 tests)
```

---

## Setup

The project uses **uv** for environment management.  All dependencies are
declared in `pyproject.toml` and pinned in `uv.lock`.

```powershell
# Install core pipeline dependencies (numpy, scipy, gensim, FAISS, PHATE, …)
uv sync

# Add word clouds in visualisation output (optional)
uv sync --extra notebook

# Add BERTopic comparison experiments (optional — large download)
uv sync --extra neural

# Add pytest + coverage for running tests (optional)
uv sync --extra dev

# Everything at once
uv sync --all-extras
```

Download NLTK data once (required for text preprocessing):

```powershell
uv run python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords')"
```

> **Note on `build.py` and `conda_requirements.txt`:** these are legacy
> artifacts from before the uv migration and should not be used.  The conda
> environment is no longer the supported setup path.

---

## Reproducing CAMSAP 2025 Results Exactly

### Step 1 — Obtain the arXiv metadata snapshot

Download `arxiv-metadata-oai-snapshot.json` from
[Kaggle arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)
(~4 GB JSONL).  The CAMSAP 2025 paper used the snapshot dated **November 2023**.

### Step 2 — Run the pipeline

Use the convenience script for your platform — it runs both variants
(with-bug and fixed) in sequence and shares the slow stages 1–2 between them:

**Windows (PowerShell):**
```powershell
.\run_experiments.ps1
```

**Linux / macOS:**
```bash
bash run_experiments.sh
```

Both scripts write progress to the console and to
`experiments/arxiv/camsap2025_with_bug/pipeline_run.log` and
`experiments/arxiv/camsap2025_fixed/pipeline_run.log`.

To run a single variant manually, use `uv run python -m legacy.run_pipeline`
directly. The pipeline defaults to the **corrected** Hellinger conversion;
add `--reproduce_legacy_bug` for exact AToMS-LP replication:

**Windows:**
```powershell
# Corrected math
uv run python -m legacy.run_pipeline `
    --input_file  arxiv-metadata-oai-snapshot.json `
    --output_dir  ./results_camsap `
    --categories  cs.LG stat.AP stat.CO stat.ME stat.ML stat.OT stat.TH `
    --year_start  2012 --year_end 2023 `
    --distance_metric hellinger --embedding_method phate `
    --linkage_method ward --cut_height 0.68

# Exact historical reproduction
uv run python -m legacy.run_pipeline `
    --input_file  arxiv-metadata-oai-snapshot.json `
    --output_dir  ./results_historical `
    --categories  cs.LG stat.AP stat.CO stat.ME stat.ML stat.OT stat.TH `
    --year_start  2012 --year_end 2023 `
    --distance_metric hellinger --embedding_method phate `
    --linkage_method ward --cut_height 0.68 `
    --reproduce_legacy_bug
```

**Linux / macOS:**
```bash
# Corrected math
uv run python -m legacy.run_pipeline \
    --input_file  arxiv-metadata-oai-snapshot.json \
    --output_dir  ./results_camsap \
    --categories  cs.LG stat.AP stat.CO stat.ME stat.ML stat.OT stat.TH \
    --year_start  2012 --year_end 2023 \
    --distance_metric hellinger --embedding_method phate \
    --linkage_method ward --cut_height 0.68

# Exact historical reproduction
uv run python -m legacy.run_pipeline \
    --input_file  arxiv-metadata-oai-snapshot.json \
    --output_dir  ./results_historical \
    --categories  cs.LG stat.AP stat.CO stat.ME stat.ML stat.OT stat.TH \
    --year_start  2012 --year_end 2023 \
    --distance_metric hellinger --embedding_method phate \
    --linkage_method ward --cut_height 0.68 \
    --reproduce_legacy_bug
```

All secondary hyperparameters (knn, LDA iterations, diffusion rate, etc.) are
read from `legacy/config.yaml`; **their defaults reproduce the CAMSAP 2025
results exactly**.  Do not change `config.yaml` unless you are intentionally
running a variant experiment.

### Step 3 — Run the BERTopic comparison

After the pipeline completes:

**Windows:**
```powershell
uv run python -m legacy.comparison.run_comparison `
    --pipeline_dir  ./results_camsap `
    --output_dir    ./results_comparison `
    --embedding_model all-MiniLM-L6-v2
```

**Linux / macOS:**
```bash
uv run python -m legacy.comparison.run_comparison \
    --pipeline_dir  ./results_camsap \
    --output_dir    ./results_comparison \
    --embedding_model all-MiniLM-L6-v2
```

Outputs:
- `results_comparison/topic_comparison_results.csv` — all metrics in one table
- `results_comparison/topic_comparison_bars.pdf` — bar chart (Figure 4 in paper)

### Expected output files

`run_experiments.ps1` (Windows) and `run_experiments.sh` (Linux/macOS) produce
two output directories.  Stages 1–2 are run once and their artifacts are
shared (copied) into both; stages 3–6 differ by the `--reproduce_legacy_bug` flag.

```
experiments/arxiv/
├── camsap2025_with_bug/         ← RUN 1 (exact AToMS-LP / CAMSAP 2025 replication)
│   ├── [all stage 1–2 artifacts, see below]
│   ├── [all stage 3–6 artifacts — with legacy bug]
│   └── pipeline_run.log
└── camsap2025_fixed/            ← RUN 2 (corrected Hellinger distances)
    ├── [stage 1–2 artifacts — copied from camsap2025_with_bug]
    ├── [all stage 3–6 artifacts — without bug]
    └── pipeline_run.log
```

**Stage 1 — Preprocessing** (`--skip_preprocess` to reuse)

| File | Contents |
|---|---|
| `main_df.pkl` | Filtered + preprocessed DataFrame (docs × metadata) |
| `id2word.pkl` | gensim `Dictionary` mapping token ID → word |
| `name_to_id.pkl` | Author name string → canonical author ID |
| `id_to_names.pkl` | Canonical author ID → list of name variants |

**Stage 2 — Ensemble LDA** (`--skip_ensemble` to reuse)

| File | Contents |
|---|---|
| `topic_models/model_chunk_0000.pkl` … `model_chunk_0143.pkl` | 144 monthly LDA models (Jan 2012 – Dec 2023) |
| `topic_vectors.pkl` | `(N_total_topics, vocab_size)` float32 — stacked phi matrices |
| `ntopics_by_chunk.pkl` | `{chunk_idx: n_topics}` — topics per month |
| `inds_by_chunk.pkl` | `{chunk_idx: [doc_ids]}` — document indices per month |
| `doc_topic_distns.pkl` | Raw per-chunk doc-topic distributions |
| `expanded_doc_topic_distns.pkl` | `{doc_id: (N_total_topics,) array}` — global-index distributions |

**Stage 3 — Distributions** (`--skip_distributions` to reuse; differs by bug flag)

| File | Contents |
|---|---|
| `coauthor_graph.pkl` | `nx.Graph` — co-author edges with paper-count weights |
| `authId_to_docs.pkl` | `{author_id: [doc_ids]}` — author → paper list |
| `knn_graph.pkl` | Topic diffusion kNN graph (knn=5, Hellinger) |
| `author_ct_distns.pkl` | `{author_id: (N_topics,) array}` — diffused author distributions |
| `doc_ct_distns.pkl` | `{doc_id: (N_topics,) array}` — diffused document distributions |

**Stage 4 — Manifold** (`--skip_manifold` to reuse; differs by bug flag)

| File | Contents |
|---|---|
| `dendrogram_Z.pkl` | scipy linkage matrix `(N_topics−1, 4)` |
| `dendrogram_heights.pkl` | `(min_h, max_h)` tuple for normalised cut height |
| `cluster_labels.pkl` | `(N_topics,)` integer meta-topic cluster assignments |
| `distance_matrix.pkl` | Full `(N_topics, N_topics)` pairwise Hellinger matrix |
| `embedding.pkl` | `(N_topics, 3)` PHATE coordinates |
| `time_labels.pkl` | `(N_topics,)` chunk index per topic (for time-coloured plot) |
| `author_embedding.pkl` | `{ids, embedding}` — author dists projected via barycentric interpolation (PHATE only) |
| `doc_embedding.pkl` | `{ids, embedding}` — doc dists projected via barycentric interpolation (PHATE only) |

`author_embedding.pkl` and `doc_embedding.pkl` are only written when
`--embedding_method phate`.  Projection uses barycentric interpolation
(`mat @ topic_embedding`), not `phate_operator.transform()`.

**Stage 5 — Scoring** (`--skip_scoring` to reuse; differs by bug flag)

| File | Contents |
|---|---|
| `encoded_root.pkl` | Root `TreeNode` of the HRG binary tree |
| `author_index_map.pkl` | `{author_id: leaf_index}` in the HRG tree |
| `author_meta_distns.pkl` | `{author_id: (N_clusters,) array}` — meta-topic distributions |
| `doc_scores.pkl` | `OrderedDict` `doc_id → entropy` (desc) |
| `link_scores.pkl` | List of `(frozenset({a1, a2}), score)` (asc by surprise) |
| `author_ranking.pkl` | `OrderedDict` `author_id → entropy` (desc) |
| `author_ranking.csv` | Human-readable author ranking (author_id, entropy) |

**Stage 6 — Visualisation** (figures saved to `figures/` subdirectory)

All filenames use the prefix `hellinger_phate_` for the CAMSAP 2025 runs.

| File | Contents |
|---|---|
| `figures/hellinger_phate_phate_cluster.pdf` | 3-D PHATE scatter coloured by meta-topic cluster |
| `figures/hellinger_phate_phate_time.pdf` | 3-D PHATE scatter coloured by time chunk |
| `figures/hellinger_phate_dendrogram.pdf` | Hierarchical dendrogram with cluster colouring |
| `figures/hellinger_phate_wordcloud_cluster_1.pdf` … `_N.pdf` | One word cloud per meta-topic cluster |
| `figures/hellinger_phate_interdiscip_docs.pdf` | Top-30 interdisciplinary documents bar chart |
| `figures/hellinger_phate_interdiscip_authors.pdf` | Top-30 interdisciplinary authors bar chart |
| `figures/hellinger_phate_coauthor_network.pdf` | Co-author network coloured by interdisciplinarity |

The number of word cloud files equals the number of meta-topic clusters
produced at `cut_height=0.68`.

### Tracking run progress

The pipeline prints timestamped INFO messages directly to the terminal as it
runs, including stage markers `[1/6] Preprocessing …` through
`[6/6] Generating figures …` and a `Saved ->` line for every artifact written.
No extra monitoring needed — just watch the terminal where the script is running.

The scripts also tee all output to `pipeline_run.log` in each output directory.
To monitor from a *second* terminal while the main one is occupied:

**Windows:**
```powershell
Get-Content -Wait experiments\arxiv\camsap2025_with_bug\pipeline_run.log
```

**Linux / macOS:**
```bash
tail -f experiments/arxiv/camsap2025_with_bug/pipeline_run.log
```

**Approximate timing** (arXiv 3.79 GB snapshot, 2012–2023, 7 categories):

| Stage | Typical time | Notes |
|---|---|---|
| 1 — Preprocessing | 5–15 min | JSON parse + NLP |
| 2 — Ensemble LDA | 2–6 hours | 144 monthly LDA models; dominant cost |
| 3 — Distributions | 10–30 min | FAISS kNN + diffusion |
| 4 — Manifold | 20–60 min | PHATE is the slow step |
| 5 — Scoring | 5–15 min | |
| 6 — Visualisation | 2–5 min | |

Stage 2 artifacts are shared between runs, so RUN 2 starts at stage 3.

### Resuming a partial run

Each stage saves its artifacts immediately.  Use `--skip_*` flags to restart
from any point without rerunning earlier stages:

**Windows:**
```powershell
uv run python -m legacy.run_pipeline [all required args] `
    --skip_preprocess `
    --skip_ensemble `
    --skip_distributions
```

**Linux / macOS:**
```bash
uv run python -m legacy.run_pipeline [all required args] \
    --skip_preprocess \
    --skip_ensemble \
    --skip_distributions
```

Available skip flags: `--skip_preprocess`, `--skip_ensemble`,
`--skip_distributions`, `--skip_manifold`, `--skip_scoring`,
`--skip_visualize`.

---

## Related Experiments (Dissertation Variants)

These use the same CLI with different primary hyperparameters.  All other
settings remain at `config.yaml` defaults.

### Different distance metrics

```powershell
# Cosine distance
uv run python -m legacy.run_pipeline ... --distance_metric cosine  --output_dir ./results_cosine

# Euclidean distance
uv run python -m legacy.run_pipeline ... --distance_metric euclidean --output_dir ./results_euclidean
```

### Different embedding methods

```powershell
uv run python -m legacy.run_pipeline ... --embedding_method umap  --output_dir ./results_umap
uv run python -m legacy.run_pipeline ... --embedding_method tsne  --output_dir ./results_tsne
uv run python -m legacy.run_pipeline ... --embedding_method pca   --output_dir ./results_pca
```

### Different dendrogram cut heights

```powershell
# Finer granularity (more meta-topics)
uv run python -m legacy.run_pipeline ... --cut_height 0.40 --output_dir ./results_fine

# Coarser granularity (fewer meta-topics)
uv run python -m legacy.run_pipeline ... --cut_height 0.90 --output_dir ./results_coarse
```

### BERTopic topic model (instead of LDA)

```powershell
uv run python -m legacy.run_pipeline ... --topic_model bertopic --output_dir ./results_bertopic
```

### Domain-specific subsets

```powershell
# Machine learning only
uv run python -m legacy.run_pipeline ... --categories cs.LG --year_start 2018 --year_end 2023

# Statistics only
uv run python -m legacy.run_pipeline ... --categories stat.ML stat.ME stat.AP
```

---

## Primary Hyperparameters

The scientifically meaningful parameters.  Pass as CLI flags.

| Parameter | CAMSAP value | Choices | Description |
|---|---|---|---|
| `input_file` | (required) | — | Path to `arxiv-metadata-oai-snapshot.json` |
| `output_dir` | (required) | — | Output directory (created if missing) |
| `categories` | cs.LG stat.AP stat.CO stat.ME stat.ML stat.OT stat.TH | any arXiv codes | Space-separated arXiv category codes |
| `year_start` | 2012 | any int | First year inclusive |
| `year_end` | 2023 | any int | Last year inclusive |
| `topic_model` | `lda` | `lda`, `bertopic` | Topic model family |
| `distance_metric` | `hellinger` | `hellinger`, `cosine`, `euclidean` | Distance for topic vectors |
| `embedding_method` | `phate` | `phate`, `umap`, `tsne`, `pca` | Manifold embedding |
| `linkage_method` | `ward` | `ward`, `complete`, `average`, `single` | Hierarchical linkage |
| `cut_height` | `0.68` | float ∈ [0, 1] | Normalised dendrogram cut height |
| `reproduce_legacy_bug` | *(off)* | flag | Use original AToMS-LP FAISS conversion (`d/√2` instead of `√(d/2)`) for exact historical reproduction. Has no effect with non-Hellinger metrics. |

---

## Mathematical Foundations

This section gives a self-contained mathematical justification for each
algorithmic stage of the pipeline.

### 1. Mixed-Membership Topic Models and the Probability Simplex

The corpus is divided into monthly temporal chunks
$\{C_1, \ldots, C_T\}$. For each chunk $C_t$, Latent Dirichlet Allocation
(LDA) estimates $K_t$ **topic-word distributions**

$$\phi_k^{(t)} \in \Delta^{V-1} = \left\{p \in \mathbb{R}^V : p_i \geq 0,\ \textstyle\sum_i p_i = 1\right\}$$

and assigns each document $d \in C_t$ a **topic distribution**
$\theta_d^{(t)} \in \Delta^{K_t - 1}$.  These are then zero-padded into the
global topic space of dimension $N = \sum_t K_t$, placing each document's
probability mass in a contiguous block corresponding to its chunk.

The resulting per-author distributions live on the
$\Delta^{N-1}$ simplex and constitute **mixed-membership vectors** in the
sense of the Mixed Membership Stochastic Block Model (MMSB; Airoldi et al.,
2008): each author $a$ is characterised by a latent membership vector
$\pi_a \in \Delta^{N-1}$, which encodes the fraction of their intellectual
activity attributable to each fine-grained topic.  The MMSB predicts the
probability of a co-authorship edge $(a, b)$ as

$$P(\text{edge} \mid a, b) = \sum_{k,\ell} \pi_a(k)\, \pi_b(\ell)\, B_{k\ell}$$

where $B$ is a $N \times N$ block-interaction matrix.  In this pipeline,
$B$ is not estimated independently from the edge data but is instead derived
from the hierarchical topic structure (Stage 5), providing a principled
prior that ties the block model to the geometry of the topic manifold.

Author mixed-membership vectors $\pi_a$ are obtained as normalised
**barycenters** — weighted averages of their documents' topic distributions on
$\Delta^{N-1}$, then smoothed by a Hellinger-distance diffusion step
(Stage 3) that propagates mass along the topic $k$-NN graph to fill in
topics that an author has not directly published in.

---

### 2. Information Geometry of Hellinger Distance

All pairwise distances between points on $\Delta^{N-1}$ are measured with
the **Hellinger distance**

$$H(p, q) = \frac{1}{\sqrt{2}}\,\|\sqrt{p} - \sqrt{q}\|_2
           = \sqrt{\frac{1}{2}\sum_{k=1}^N \!\left(\sqrt{p_k} - \sqrt{q_k}\right)^2}
\;\in [0, 1].$$

Its information-geometric interpretation is as follows.  The probability
simplex $\Delta^{N-1}$ is a **statistical manifold** equipped with the
**Fisher–Rao metric**

$$g_{ij}(p) = \frac{\delta_{ij}}{p_i}.$$

The square-root map $\Phi : \Delta^{N-1} \to S^{N-1}_+$, $p \mapsto \sqrt{p}$,
is an isometric embedding of $(\Delta^{N-1}, g)$ into the positive orthant of
the $(N-1)$-sphere with the round metric.  Under this embedding, the **chord
distance** between $\Phi(p)$ and $\Phi(q)$ is exactly $\|\sqrt{p} - \sqrt{q}\|_2$,
so

$$H(p, q) = \frac{1}{\sqrt{2}}\|\Phi(p) - \Phi(q)\|_2$$

is the Euclidean chord distance in the square-root representation, normalised
to $[0,1]$.  Equivalently, $H(p,q) = \sqrt{1 - \mathrm{BC}(p,q)}$ where
$\mathrm{BC}(p,q) = \sum_k \sqrt{p_k q_k}$ is the **Bhattacharyya
coefficient**, and the Fisher–Rao geodesic arc length is
$d_{\mathrm{FR}}(p,q) = 2\arccos\bigl(\mathrm{BC}(p,q)\bigr) \approx 2H$
for nearby distributions.

Because $H$ arises from an underlying Euclidean norm on $\sqrt{p}$ vectors,
it supports Ward linkage and PHATE diffusion with full geometric validity —
the two algorithmic choices justified below.

---

### 3. Ward Linkage on the Statistical Manifold

The **Ward hierarchical clustering** at Stage 4 operates directly on the
$N \times N$ Hellinger distance matrix among topic vectors.

Ward's criterion merges the pair of clusters $(A, B)$ that minimises the
increase in total **within-cluster sum of squares**:

$$\Delta(A, B) = \frac{|A|\,|B|}{|A| + |B|} H^2(\mu_A, \mu_B)$$

where $\mu_C$ denotes the Fréchet mean (centroid) of cluster $C$.  Under
Hellinger distance, the centroid is the point $\mu_C$ minimising
$\sum_{p \in C} H^2(p, \mu_C)$, which in the square-root representation
reduces to the ordinary Euclidean centroid $\bar{v}_C = \frac{1}{|C|}\sum_{p \in C} \sqrt{p}$,
renormalised to the simplex.

Because $H(p,q) = \frac{1}{\sqrt 2}\|\sqrt p - \sqrt q\|_2$ is a Euclidean
distance (up to the constant $1/\sqrt{2}$), the **Lance–Williams update** is
valid:

$$d(A \cup B,\, C)^2 = \frac{|A|+|C|}{|A|+|B|+|C|} d(A,C)^2
                     + \frac{|B|+|C|}{|A|+|B|+|C|} d(B,C)^2
                     - \frac{|C|}{|A|+|B|+|C|} d(A,B)^2.$$

Geometrically, Ward linkage partitions the topic simplex into **maximally
compact, well-separated meta-topic clusters**, each representing a coherent
research community.  In the corrected MSTML implementation, merge heights
are in true Hellinger units, so the normalised cut height parameter
`cut_height` ∈ [0, 1] has a direct information-geometric meaning: it
selects the granularity at which topic distributions are considered distinct
communities.

> **AToMS-LP vs. MSTML (legacy reproduction note).** AToMS-LP's
> dendrogram-building code has an additional bug on top of the basic FAISS
> conversion error: it stores `sq_l2²` (≡ 4H⁴) instead of `H` in the
> condensed distance matrix (see the [FAISS Hellinger Bug Fix](#faiss-hellinger-bug-fix)
> section for the full derivation).  Pass `legacy_bug=True` to
> `build_topic_dendrogram` to reproduce AToMS-LP's dendrogram exactly;
> the default (`legacy_bug=False`) uses correct Hellinger distances.

---

### 4. Hellinger–PHATE and Diffusion on the Topic Manifold

To embed the $N$ topic distributions into a low-dimensional coordinate
system, the pipeline uses **PHATE** (Potential of Heat-diffusion for
Affinity-based Transition Embedding; Moon et al., 2019) with the precomputed
Hellinger distance matrix as input.

PHATE builds an affinity matrix

$$K_{ij} = \exp\!\left(-\frac{H(p_i, p_j)^2}{\sigma_i^2}\right)$$

where $\sigma_i$ is a local bandwidth estimated from the $k$-nearest
neighbours of $p_i$.  Row-normalising $K$ yields the **Markov diffusion
operator** $P$; the $(i,j)$ entry of $P^t$ is the probability of reaching
$p_j$ from $p_i$ by a $t$-step random walk on the topic graph.  PHATE then
forms the **diffusion potential**

$$U_{ij}^{(t)} = -\log P^t_{ij}$$

and embeds via multi-dimensional scaling (MDS) on the resulting potential
distances, producing coordinates that simultaneously preserve local neighbourhood
structure and global topological relationships.

The information-geometric justification for using Hellinger as the input
metric is twofold.  First, because $H$ derives from the Fisher–Rao metric, it
is **invariant to reparametrisation** of the probability simplex: equal
Hellinger distances correspond to equal distinguishability under any statistical
test, not just squared-Euclidean displacement.  Second, the square-root
embedding $\Phi(p) = \sqrt{p}$ places the simplex on a sphere, whose curved
geometry is naturally captured by the diffusion operator's multi-scale
random walk — PHATE's $t$ parameter effectively integrates over the spectrum
of the Laplace–Beltrami operator on this manifold.

The result is a three-dimensional **topic manifold** in which Euclidean
proximity reflects shared information-theoretic content, and the temporal
trajectory of topics can be read from the time-coloured embedding.  Author
and document distributions are projected into the same space via
$\operatorname{phate.transform}(D_{\text{cross}})$, where $D_{\text{cross}}$
is the $(n_{\text{new}} \times N)$ cross-Hellinger distance matrix from new
distributions to the training topics.

---

### 5. Hierarchical Random Graph Link Inference

The interdisciplinarity scores are grounded in the **Hierarchical Random
Graph** (HRG) framework of Clauset, Moore & Newman (2008).

Given a dendrogram $\mathcal{D}$ over $N$ topics, every internal node $r$
defines a bipartition of the leaves into a left subtree $L_r$ and a right
subtree $R_r$.  Under the HRG model, each co-authorship edge $(a,b)$
independently exists with probability $p_r$, where $r$ is the lowest common
ancestor (LCA) of the meta-topics predominantly occupied by $a$ and $b$.
The maximum likelihood estimate of $p_r$ is

$$\hat{p}_r = \frac{E_r}{|L_r|\,|R_r|}$$

where $E_r$ is the number of co-author edges whose endpoints fall on opposite
sides of the split at $r$.

Because authors have **mixed membership** (their mass is spread across many
topics rather than assigned to exactly one), the pipeline uses expected
edge counts.  Defining the probability that author $a$ belongs to the left
subtree at node $r$ as $\ell_r(a) = \sum_{k \in L_r} \pi_a(k)$, the
contribution of edge $(a,b)$ to $E_r$ is

$$\ell_r(a)\bigl(1 - \ell_r(b)\bigr) + \ell_r(b)\bigl(1 - \ell_r(a)\bigr)$$

and the denominator generalises to
$\hat{p}_r = \mathrm{numerator} / \bigl(\mathbb{E}[|L_r|]\,\mathbb{E}[|R_r|]\bigr)$.

The **link likelihood score** for a co-author pair $(a, b)$ is then

$$s(a, b) = \sum_{k,\ell} \pi_a(k)\,\pi_b(\ell)\,\mathrm{LPM}_{k\ell}$$

where $\mathrm{LPM}_{k\ell} = \hat{p}_{\mathrm{LCA}(k,\ell)}$ is the
link-probability matrix entry for meta-topics $k$ and $\ell$.  This is
precisely the MMSB edge probability $\pi_a^{\top} B\, \pi_b$ with
$B = \mathrm{LPM}$, closing the loop from Stage 1.

**Low** $s(a,b)$ means the collaboration is *surprising* given the topic
hierarchy — the two authors occupy distant branches of the dendrogram —
which operationalises interdisciplinarity as anomaly under the fitted
hierarchical block model.

**Author interdisciplinarity** is the **Shannon entropy** of the
distribution $\pi_a$ over meta-topics at the chosen cut height:

$$\mathcal{I}(a) = H(\pi_a) = -\sum_{k=1}^{M} \pi_a(k)\,\log_2 \pi_a(k)$$

where $M$ is the number of meta-topics.  The entropy is maximised at
$\log_2 M$ bits when $\pi_a$ is uniform — an author equally active in all
meta-topics — and zero when $\pi_a$ is a point mass (fully specialised).

---

## Secondary Hyperparameters (`config.yaml`)

Stable algorithmic settings.  Override by passing `--config my_override.yaml`.

```yaml
pipeline:
  months_per_chunk: 1          # Temporal window size (1 = monthly chunks)
  max_authors_per_doc: 20      # Discard documents with more authors
  author_disambig_threshold: 0.90  # TF-IDF char-trigram cosine threshold

text_processing:
  vocab_low_freq_thresh: 1
  vocab_high_freq_frac: 0.995
  vocab_lda:
    lambda: 0.6                # LDAvis relevance λ for vocabulary reduction
    top_n: 2000                # Vocabulary size after LDA-guided reduction
    num_topics: 50             # Topics for initial vocabulary-reduction LDA

topic_models:
  lda:
    docs_per_topic: 100        # Heuristic: num_topics = chunk_size / docs_per_topic
    ngibbs: 50
    npasses: 5
    smoothing_gamma: 0.75      # Exponential decay γ for neighbouring-chunk augmentation

diffusion:
  knn: 5                       # k for topic diffusion graph
  rate: 0.7                    # Diffusion rate α
  num_iterations: 1

dendrogram:
  knn: 100                     # k for FAISS approximate dendrogram graph

embeddings:
  phate:
    n_components: 3
    gamma: 0.0
    knn: 5
    t: auto
```

---

## Performance Notes

### `_tree.py` — HRG Tree Encoding

The AToMS-LP Cython source (`fast_encode_tree.pyx`) is included in
`legacy/fast_encode_tree.pyx`.  Build it once for full Cython speed:

```powershell
# From the repo root (cython and numpy are already installed via uv sync)
uv run python legacy/setup.py build_ext --inplace
```

`_tree.py` automatically imports the compiled extension when present; it falls
back to a numpy-vectorised pure-Python implementation otherwise.  Both paths
produce identical numerical results.  For the full arXiv corpus (thousands of
topics, tens of thousands of authors), the compiled extension is substantially
faster for the tree-encoding stage.

---

## FAISS Hellinger Bug Fix

### Background: FAISS IndexFlatL2 on square-root vectors

To compute Hellinger distances via FAISS, topic distributions $p$ are
mapped to their square-root representations $v = \sqrt{p}$ before indexing.
`IndexFlatL2` then returns the squared Euclidean distance in that space:

$$\text{sq\_l2} = \|\sqrt{p} - \sqrt{q}\|^2 = 2\,H(p,q)^2$$

So the true Hellinger distance is:

$$H(p,q) = \sqrt{\frac{\text{sq\_l2}}{2}}$$

### Bug 1 — Basic conversion error (affects all AToMS-LP distance calls)

AToMS-LP converts the raw FAISS output with:

```python
distances = sq_l2 / np.sqrt(2)    # WRONG: gives H²·√2, not H
```

**What this produces:** $\frac{\text{sq\_l2}}{\sqrt{2}} = \frac{2H^2}{\sqrt{2}} = H^2\sqrt{2}$

That is neither $H$ nor $H^2$ — it is a scaled version of $H^2$.

MSTML fixes this everywhere with:

```python
distances = np.sqrt(sq_l2 / 2.0)  # CORRECT: gives true H ∈ [0, 1]
```

This fix is implemented in `_math.py:faiss_sq_l2_to_hellinger` and
`_math.py:faiss_sq_l2_to_distance`, and verified by
`test_math.py::TestFaissHellingerConversion`.

### Bug 2 — Ward's dendrogram: double-conversion produces sq_l2² (AToMS-LP-specific)

The AToMS-LP dendrogram-building cell applies an *additional* operation in
the Ward's linkage branch that compounds Bug 1 into a much larger error:

```python
# Step 1 (Bug 1): convert FAISS output — intending H, but getting H²√2
distances = sq_l2 / np.sqrt(2)           # result: H²√2 = sq_l2/√2

# Step 2 (Ward branch): undo the conversion to recover sq_l2
adjusted_distances = distances * np.sqrt(2)  # result: sq_l2

# Step 3: square the result to store in the condensed matrix
condensed_matrix[i, j] = adjusted_distances ** 2  # result: sq_l2²
```

**What AToMS-LP stores:**
$$\text{sq\_l2}^2 = \bigl(\|\sqrt{p}-\sqrt{q}\|^2\bigr)^2 = (2H^2)^2 = 4H^4$$

**What scipy's `linkage(..., method='ward')` expects:**
Scipy takes input distances $d$, then internally computes $d^2$ as the
Ward merge cost.  The correct input is $d = H$, so scipy uses $H^2$ — the
geometrically valid criterion.  AToMS-LP instead passes $d = \text{sq\_l2}^2 = 4H^4$,
so scipy uses $d^2 = 16H^8$, yielding a dendrogram whose merge heights have
no direct relationship to Hellinger distance.

**MSTML fixed implementation** passes $H = \sqrt{\text{sq\_l2}/2}$ directly
to scipy, so all dendrogram merge heights are in true Hellinger units.

**Controlled by flag:** pass `legacy_bug=True` to `build_topic_dendrogram`
to reproduce AToMS-LP's condensed matrix (stores sq_l2²) for exact
numerical comparison; the default `legacy_bug=False` uses correct $H$.

### Summary table

| Context | AToMS-LP computes | MSTML (fixed) computes |
|---|---|---|
| Diffusion graph kNN distances | $H^2\sqrt{2}$ (Bug 1) | $H = \sqrt{\text{sq\_l2}/2}$ |
| Dendrogram (non-Ward) | $H^2\sqrt{2}$ (Bug 1) | $H$ |
| Dendrogram (Ward) | $\text{sq\_l2}^2 = 4H^4$ (Bugs 1+2) | $H$ |
| Scipy Ward merge cost | $16H^8$ | $H^2$ |

---

## Running the Tests

All tests live in `legacy/tests/` and are self-contained — no arXiv file, no
network access, no GPU required.  Every test uses synthetic data generated by
`conftest.py` fixtures.

```bash
# Run the entire legacy test suite from the repo root
uv run pytest legacy/tests/ -v

# Run a single file
uv run pytest legacy/tests/test_math.py -v

# Run with coverage report
uv run pytest legacy/tests/ --cov=legacy --cov-report=term-missing
```

Expected result: **201 passed** across all four test files.

---

## Test Reference

### `test_math.py` — `_math.py` unit tests (42 tests)

#### `TestHellinger`
| Test | What it checks |
|---|---|
| `test_range` | H(p,q) ∈ [0, 1] for 20 random pairs |
| `test_symmetry` | H(p,q) = H(q,p) |
| `test_self_distance_is_zero` | H(p,p) < 1e-7 |
| `test_orthogonal_distributions_equal_one` | Disjoint-support distributions give H = 1 |
| `test_matches_definition` | Result equals `√(½ Σ (√pᵢ − √qᵢ)²)` directly |
| `test_triangle_inequality` | H satisfies H(p,r) ≤ H(p,q) + H(q,r) |

#### `TestHellingerMatrix`
| Test | What it checks |
|---|---|
| `test_shape` | Output is (n, n) square |
| `test_diagonal_zeros` | Self-distances are zero |
| `test_symmetric` | M = Mᵀ |
| `test_range` | All entries in [0, 1] |
| `test_matches_elementwise_hellinger` | M[i,j] matches scalar `hellinger(X[i], X[j])` |
| `test_single_row_gives_zeros` | 1×1 matrix has value 0 |

#### `TestFaissHellingerConversion`
| Test | What it checks |
|---|---|
| `test_corrected_formula_matches_direct` | `sqrt(sq_l2/2)` matches scalar `hellinger()` for 50 pairs |
| `test_old_formula_is_wrong` | `D/√2` (original AToMS-LP) is more wrong than `√(D/2)` |
| `test_output_range_for_probability_vectors` | Converted values lie in [0, 1] |
| `test_zero_distance_stays_zero` | Input 0 → output 0 |
| `test_dispatch_hellinger_alias` | `faiss_sq_l2_to_distance(sq, "hellinger")` routes correctly |
| `test_dispatch_cosine` | `faiss_sq_l2_to_distance(sq, "cosine")` returns `sq/2` |
| `test_dispatch_euclidean` | `faiss_sq_l2_to_distance(sq, "euclidean")` returns `√sq` |
| `test_dispatch_unknown_metric_raises` | Unknown metric raises ValueError or KeyError |

#### `TestPrepareVectors`
| Test | What it checks |
|---|---|
| `test_hellinger_returns_sqrt` | Hellinger prep returns `√X` |
| `test_hellinger_nonneg` | All values ≥ 0 after sqrt |
| `test_cosine_unit_norm` | Cosine prep L2-normalises each row |
| `test_euclidean_passthrough` | Euclidean prep is identity |
| `test_output_dtype_float32` | Always returns float32 for FAISS compatibility |

#### `TestEntropy`
| Test | What it checks |
|---|---|
| `test_uniform_is_maximum_over_same_support` | Uniform distribution maximises entropy |
| `test_one_hot_is_zero` | Point mass has entropy 0 |
| `test_binary_fair_coin_equals_one_bit` | H([0.5, 0.5]) = 1 |
| `test_uniform_n_value` | H(uniform over n) = log₂(n) |
| `test_no_nan_for_zero_entries` | Zero-probability entries handled without NaN |
| `test_nonneg` | Entropy ≥ 0 for 20 random distributions |
| `test_monotone_in_uniformity` | More-spread distribution has higher entropy |

#### `TestTermRelevance`
| Test | What it checks |
|---|---|
| `test_output_shape` | Returns (vocab_size,) array |
| `test_finite_output` | No NaN or Inf |
| `test_lambda_1_ranks_by_phi` | At λ=1, top-φ word ranks in top-3 |
| `test_lambda_0_penalises_common_words` | At λ=0, topic-specific rare words score higher |
| `test_various_lambda_finite[0.0/0.5/1.0]` | Finite output at boundary λ values |

#### `TestCorpusWordProbabilities`
| Test | What it checks |
|---|---|
| `test_sums_to_one` | All word probabilities sum to 1 |
| `test_covers_all_words` | Every word in corpus has an entry |
| `test_relative_frequencies` | Frequency ratios are correct |
| `test_empty_docs_ignored` | Empty documents do not affect result |

#### `TestDiffusionWeights`
| Test | What it checks |
|---|---|
| `test_formula_w_ij_equals_1_minus_d_over_sum` | Weights match `1 − dᵢ / Σd` |
| `test_closer_neighbor_gets_higher_weight` | Smaller distance → larger weight |
| `test_single_neighbor_degenerates_to_zero` | k=1 gives weight 0 (caller guards with w=1) |
| `test_nonneg_with_multiple_neighbors` | All weights ≥ 0 |
| `test_uniform_distances_give_zero_weights` | Equal distances → all weights 0 |

---

### `test_pipeline.py` — pipeline integration tests (87 tests)

#### `TestConfig`
| Test | What it checks |
|---|---|
| `test_load_default_pipeline_config` | `config.yaml` loads; required fields present and positive |
| `test_lda_section_defaults` | LDA defaults: docs_per_topic=100, ngibbs=50, smoothing_gamma=0.75 |
| `test_embeddings_phate_defaults` | PHATE defaults: n_components=3, gamma=0.0 |
| `test_primary_config_defaults` | PrimaryConfig default values are set |
| `test_primary_config_validates_year_range` | year_start > year_end raises ValueError |
| `test_topic_models_get_kwargs_lda` | `get_kwargs("lda")` returns expected dict |
| `test_embeddings_get_kwargs_phate` | `get_kwargs("phate")` returns expected dict |
| `test_config_yaml_override` | Custom YAML file overrides individual values |

#### `TestTemporalChunking`
| Test | What it checks |
|---|---|
| `test_chunks_count` | Correct number of month-start chunks |
| `test_chunks_non_empty` | No empty chunks produced |
| `test_chunks_cover_all_docs` | All document IDs appear in exactly one chunk |
| `test_chunk_dates_ordered` | Chunk dates increase monotonically |

#### `TestExpandDistributions`
| Test | What it checks |
|---|---|
| `test_output_length` | One expanded vector per document |
| `test_vector_length` | Each vector has length N_TOPICS_TOTAL |
| `test_in_chunk_block_sums_to_one` | The block for the document's own chunk sums to 1 |
| `test_out_of_chunk_blocks_zero` | All other chunks' blocks are exactly zero |
| `test_dtype_float32` | Output is float32 |
| `test_expand_distributions_function` | `expand_distributions()` end-to-end call |

#### `TestLDATopicModel`
| Test | What it checks |
|---|---|
| `test_fit_runs` | `LDATopicModel.fit()` completes without error |
| `test_topic_vectors_shape_and_normalised` | phi is (n_topics, vocab_size), rows sum to 1 |
| `test_doc_topic_distributions_shapes` | theta is (n_docs, n_topics), rows sum to 1 |
| `test_num_topics_min_four` | Very small chunks still produce ≥ 4 topics |

#### `TestCoauthorGraph`
| Test | What it checks |
|---|---|
| `test_node_coverage` | Every author appearing in the data is a graph node |
| `test_edge_weights_positive` | Co-authorship edge weights > 0 |
| `test_no_self_loops` | No author connected to themselves |

#### `TestAuthorBarycenters`
| Test | What it checks |
|---|---|
| `test_all_authors_covered` | Every author in the corpus has a barycenter |
| `test_distributions_sum_to_one` | Each barycenter sums to 1 |
| `test_vector_length` | Each barycenter has length N_TOPICS_TOTAL |
| `test_nonneg` | All barycenter values ≥ 0 |

#### `TestDiffusionGraph`
| Test | What it checks |
|---|---|
| `test_node_count` | Graph has exactly N_TOPICS_TOTAL nodes |
| `test_edge_weights_positive` | All edge weights > 0 |
| `test_knn_bound_on_degree` | Every node has ≥ 1 neighbour |
| `test_distances_in_unit_interval_for_hellinger` | Hellinger edge distances in [0, 1] |

#### `TestDiffusion`
| Test | What it checks |
|---|---|
| `test_output_keys_match_input` | Same author keys in and out |
| `test_distributions_normalised` | Diffused distributions sum to 1 |
| `test_nonneg` | All diffused values ≥ 0 |
| `test_vector_length` | Vectors have correct length |
| `test_single_neighbour_guard` | Single-neighbour nodes diffuse correctly (no zero-weight collapse) |

#### `TestPairwiseDistances`
| Test | What it checks |
|---|---|
| `test_shape` | Output is (N_TOPICS_TOTAL, N_TOPICS_TOTAL) |
| `test_symmetric` | D = Dᵀ |
| `test_diagonal_zero` | D[i,i] = 0 |
| `test_range` | All entries in [0, 1] for Hellinger |
| `test_other_metrics_nonneg_and_symmetric[cosine]` | Cosine distances ≥ 0, symmetric |
| `test_other_metrics_nonneg_and_symmetric[euclidean]` | Euclidean distances ≥ 0, symmetric |

#### `TestTopicDendrogram`
| Test | What it checks |
|---|---|
| `test_linkage_matrix_shape` | scipy linkage is (N-1, 4) |
| `test_heights_nonneg` | All merge heights ≥ 0 |
| `test_heights_monotone` | Heights are non-decreasing (valid dendrogram) |
| `test_first_height_in_unit_interval` | First (smallest) merge ≤ 1.0 (input is Hellinger) |
| `test_min_le_max` | min_height ≤ max_height |
| `test_cluster_count_range` | Merge count = N_TOPICS_TOTAL − 1 |

#### `TestCutDendrogram`
| Test | What it checks |
|---|---|
| `test_labels_integer_array[0.0/0.25/0.5/0.75/1.0]` | Labels are integer array at each cut height |
| `test_cut_0_gives_single_cluster` | Cut at bottom → 1 cluster |
| `test_cut_1_gives_max_clusters` | Cut at top → maximum clusters |
| `test_more_clusters_at_lower_cut` | Lower cut → more clusters than higher cut |
| `test_out_of_range_raises` | cut_height outside [0, 1] raises an error |

#### `TestComputeEmbedding`
| Test | What it checks |
|---|---|
| `test_phate_shape` | PHATE output is (N_TOPICS_TOTAL, 3); skipped if phate not installed |
| `test_pca_shape` | PCA output has correct shape |
| `test_tsne_shape` | t-SNE output has correct shape |
| `test_unknown_method_raises` | Unknown method name raises ValueError |

#### `TestTreeNode`
| Test | What it checks |
|---|---|
| `test_leaf_is_leaf` | Leaf node (no children) reports `is_leaf() == True` |
| `test_internal_not_leaf` | Internal node with children is not a leaf |
| `test_leaf_count` | `get_leaf_count()` returns correct count |
| `test_get_all_leaf_ids` | `get_all_leaf_ids()` returns complete set |

#### `TestEncodeTree`
| Test | What it checks |
|---|---|
| `test_root_is_tree_node` | Root is a TreeNode instance |
| `test_root_is_not_leaf` | Root of an N>1 tree is not a leaf |
| `test_author_index_map_coverage` | Every author in input has an index |
| `test_author_probs_shape` | Author probability matrix is (n_authors, n_meta_topics) |
| `test_root_original_leaf_ids_complete` | Root leaf IDs cover all topics |
| `test_link_prob_in_unit_interval` | Every internal node's link probability ∈ [0, 1] |

#### `TestAuthorMetaDistributions`
| Test | What it checks |
|---|---|
| `test_all_authors_present` | Every author has a meta-topic distribution |
| `test_distributions_sum_to_one` | Each distribution sums to 1 |
| `test_nonneg` | All entries ≥ 0 |

#### `TestInterdisciplinarityScoring`
| Test | What it checks |
|---|---|
| `test_doc_scores_nonneg` | Document entropy scores ≥ 0 |
| `test_doc_scores_ordered_descending` | Scores sorted highest-first |
| `test_author_ranking_ordered_descending` | Author ranking sorted highest-first |
| `test_link_scores_ordered_ascending` | Link scores sorted lowest-first (most likely first) |
| `test_author_ranking_covers_all` | Every author in coauthor graph is ranked |
| `test_entropy_bounded_by_log2_n_meta_topics` | No score exceeds theoretical maximum |

#### `TestEndToEnd`
| Test | What it checks |
|---|---|
| `test_full_pipeline_runs_to_completion` | Build dendrogram → cut → score on synthetic data; validates Z shape, label count, all result keys, and score ordering |

---

### `test_comparison.py` — comparison experiment tests (59 tests)

#### `TestPairwiseHellinger`
| Test | What it checks |
|---|---|
| `test_self_distance_is_zero` | H(X, X) = 0 for all rows |
| `test_symmetry` | `_pairwise_hellinger(X,Y)` = `_pairwise_hellinger(Y,X)ᵀ` |
| `test_orthogonal_distributions_equal_one` | Disjoint supports give distance 1 |
| `test_range` | All pairwise distances in [0, 1] |
| `test_output_shape` | Returns (n, m) matrix for (n, d) and (m, d) inputs |
| `test_matches_scalar_formula` | Matches `_math.hellinger()` for a single pair |

#### `TestComputeAlignmentScores`
| Test | What it checks |
|---|---|
| `test_returns_expected_keys` | Result dict has `mean_distance`, `skewness`, `n_pairs`, `all_distances` |
| `test_n_pairs_correct` | n_pairs = (n_chunks − 1) × n_topics × knn |
| `test_perfect_alignment_zero_mean_distance` | Identical chunks across time → mean distance < 1e-4 |
| `test_random_chunks_positive_distance` | Independent random chunks → mean distance > 0 |
| `test_distances_in_unit_interval_hellinger` | Hellinger distances in [0, 1] |
| `test_cosine_metric_non_negative` | Cosine alignment distances ≥ 0 |
| `test_euclidean_metric_non_negative` | Euclidean alignment distances ≥ 0 |
| `test_single_chunk_gives_zero_pairs` | One chunk → no consecutive pairs → NaN mean |
| `test_knn_gt_one_multiplies_pairs` | knn=2 doubles the pair count |
| `test_empty_chunk_skipped_gracefully` | Zero-row chunks skipped without error |
| `test_fixture_chunks_mean_distance_finite` | Fixture chunks produce finite mean distance |

#### `TestDiversityAtK`
| Test | What it checks |
|---|---|
| `test_all_unique_words_gives_one` | Fully non-overlapping topics → Diversity@k = 1 |
| `test_all_identical_words` | All topics share same words → Diversity = 1/n_topics |
| `test_range` | Score always in [0, 1] |
| `test_empty_list_gives_zero` | Empty topic list → 0 |
| `test_k_larger_than_words_uses_all_words` | k exceeding topic length uses available words |
| `test_single_topic` | One topic → Diversity = 1 (all words trivially unique) |
| `test_partial_overlap` | Known partial overlap gives exact expected score |

#### `TestMeanPairwiseHellinger`
| Test | What it checks |
|---|---|
| `test_single_row_returns_nan` | Single topic → NaN (no pairs) |
| `test_identical_rows_give_zero` | Copies of same vector → mean distance ≈ 0 |
| `test_orthogonal_topics_give_one` | Disjoint-support topics → mean = 1 |
| `test_range` | Mean pairwise Hellinger in [0, 1] |
| `test_monotone_separation` | Near-duplicate topics < fully-random topics |

#### `TestMeanPairwiseCosineDistance`
| Test | What it checks |
|---|---|
| `test_single_row_returns_nan` | Single topic → NaN |
| `test_identical_rows_give_zero` | Identical rows → mean ≈ 0 |
| `test_range` | Cosine distance in [0, 2] for non-negative vectors |
| `test_orthogonal_vectors` | Orthogonal unit vectors → distance = 1 |

#### `TestComputeDiversity`
| Test | What it checks |
|---|---|
| `test_returns_expected_keys` | Result has `diversity_at_k`, `mean_hellinger`, `mean_cosine` |
| `test_no_vectors_gives_nan_distance_metrics` | Without topic vectors, vector metrics are NaN |
| `test_with_vectors_gives_finite_distances` | With topic vectors, both distance metrics finite |
| `test_diversity_at_k_value_matches_standalone` | Result matches `diversity_at_k()` directly |

#### `TestComputeEnsembleDiversity`
| Test | What it checks |
|---|---|
| `test_returns_expected_keys` | All three metric keys present |
| `test_averaging_across_chunks` | Correctly averages per-chunk diversity values |
| `test_nan_chunks_excluded` | Single-topic chunks (NaN Hellinger) excluded from average |
| `test_no_vectors_all_nan` | No vectors → both distance metrics are NaN |
| `test_fixture_chunks_finite_diversity` | Fixture chunks produce finite diversity@k and mean Hellinger |

#### `TestComputeCoherence`
| Test | What it checks |
|---|---|
| `test_returns_dict_with_expected_measures` | Keys `c_v`, `c_npmi`, `c_uci` all present |
| `test_values_are_float` | All values are Python floats |
| `test_empty_topics_returns_nan` | Empty topic list → NaN for all measures |
| `test_topics_after_trim_to_top_n` | `top_n` trimming does not raise |
| `test_custom_measures_subset` | Only requested measures are computed |

#### `TestComputeEnsembleCoherence`
| Test | What it checks |
|---|---|
| `test_returns_dict_with_measures` | Output is a float dict |
| `test_averaging_produces_finite_or_nan` | All values are floats (finite or NaN) |
| `test_single_chunk_equals_per_chunk_value` | Ensemble of 1 chunk = single-chunk value |

#### `TestBERTopicEmbed`
| Test | What it checks |
|---|---|
| `test_train_returns_correct_number_of_models` | Returns one model per input chunk |
| `test_phi_rows_sum_to_one_or_zero` | Each topic vector is normalised (sum=1) or all-zero |
| `test_get_top_words_structure` | Per-chunk per-topic word lists have ≤ top_n words |
| `test_phi_shape_matches_vocab` | phi columns = gensim vocabulary size |

#### `TestMSTMLvsRandomBaseline` — structural CAMSAP comparison
| Test | What it checks |
|---|---|
| `test_identical_chunks_lower_bound` | Perfect temporal stability → alignment distance < 1e-4 |
| `test_random_chunks_positive_distance` | Fully random topics → alignment distance > 0.01 |
| `test_smoothed_chunks_less_distance_than_random` | Temporally-smoothed topics align better than fully independent random topics (validates MSTML's key claim) |
| `test_diversity_specialised_vs_uniform_topics` | Specialised topics have higher Diversity@k than all-identical topics |
| `test_hellinger_lower_than_random_for_stable_ensemble` | Near-duplicate topics have lower mean Hellinger than fully random topics |

---

### `test_integration.py` — `MstmlOrchestrator` regression tests (13 tests)

These test the `mstml/` library implementation of the same pipeline stages,
using a synthetic injected `MstmlOrchestrator` state.

#### `TestFaissHellingerConversion`
| Test | What it checks |
|---|---|
| `test_faiss_squared_l2_to_hellinger` | `sqrt(D/2)` matches direct Hellinger for all 15 synthetic topics |
| `test_old_formula_would_be_wrong` | `D/√2` has systematically higher error than `√(D/2)` |

#### `TestBuildAuthorDocumentDistributions`
| Test | What it checks |
|---|---|
| `test_expanded_distributions_shape` | Each doc gets a (N_chunks × N_topics_per_chunk,) vector summing to 1 |
| `test_author_barycenters_shape` | Each author barycenter has the same length and is non-zero |
| `test_all_authors_covered` | Every author in the doc corpus has a barycenter |

#### `TestApplyDiffusion`
| Test | What it checks |
|---|---|
| `test_diffused_distributions_shape` | Diffused vectors have correct length |
| `test_diffused_distributions_normalized` | Diffused distributions sum to 1 |
| `test_diffusion_matrix_shape` | `diffusion_matrix` is (N_total_topics, N_total_topics) |
| `test_diffusion_preserves_known_entries` | Non-zero entries in the original remain non-zero after diffusion |

#### `TestBuildTopicManifold`
| Test | What it checks |
|---|---|
| `test_linkage_matrix_shape` | Linkage matrix is (N_topics − 1, 4) |
| `test_dendrogram_heights_are_non_negative` | All heights ≥ 0; first merge height ≤ 1.0 (input is Hellinger) |
| `test_heights_monotonically_non_decreasing` | Heights are non-decreasing (required for valid linkage) |
| `test_min_max_cut_heights_stored` | `min_cut_height ∈ [0,1]` and `min ≤ max` |

---

## Python API

```python
from legacy._config       import PrimaryConfig, load_pipeline_config
from legacy._preprocessing import load_arxiv_jsonl, preprocess_text, disambiguate_authors
from legacy._topic_models  import create_temporal_chunks, train_ensemble, expand_distributions
from legacy._distributions import (build_coauthor_graph, compute_author_barycenters,
                                    build_diffusion_graph, diffuse_distributions)
from legacy._manifold      import (compute_pairwise_distances, build_topic_dendrogram,
                                    cut_dendrogram, compute_embedding,
                                    project_distributions_onto_embedding)
from legacy._scoring       import run_scoring
from legacy._visualization import plot_phate_embedding, generate_meta_topic_wordclouds

# Load config
cfg     = load_pipeline_config()           # reads legacy/config.yaml
primary = PrimaryConfig(
    input_file="arxiv-metadata-oai-snapshot.json",
    output_dir="./results",
    categories=["cs.LG", "stat.ML"],
    year_start=2018, year_end=2023,
)

# Stage 1 — preprocessing
df, id2word = preprocess_text(
    load_arxiv_jsonl(primary.input_file, primary.categories,
                     primary.year_start, primary.year_end),
    cfg,
)
df, name_to_id, id_to_names = disambiguate_authors(df)

# Stage 2 — topic ensemble
chunks = create_temporal_chunks(df)
models, topic_vectors, ntopics_by_chunk, inds_by_chunk = train_ensemble(
    chunks, id2word, "lda", cfg, output_dir="./results/topic_models"
)
expanded = expand_distributions(
    {doc_id: ... for doc_id, dist in ...}, inds_by_chunk, ntopics_by_chunk
)

# Stage 3 — distributions
coauthor_graph     = build_coauthor_graph(df)
author_barycenters, *_ = compute_author_barycenters(expanded, df)
knn_graph          = build_diffusion_graph(topic_vectors, knn=cfg.diffusion.knn)
author_ct_distns   = diffuse_distributions(knn_graph, author_barycenters)

# Stage 4 — manifold
Z, min_h, max_h = build_topic_dendrogram(topic_vectors)
cluster_labels  = cut_dendrogram(Z, cut_height=0.68, min_height=min_h, max_height=max_h)
dist_matrix     = compute_pairwise_distances(topic_vectors)
embedding, phate_op = compute_embedding(dist_matrix, "phate")
# phate_op is the fitted PHATE operator (None for umap/tsne/pca)

# Project author/doc distributions into PHATE space (PHATE only)
if phate_op is not None:
    author_vecs = np.array(list(author_ct_distns.values()))
    author_emb  = project_distributions_onto_embedding(
        topic_vectors, author_vecs, phate_op)  # (n_authors, 3)

# Stage 5 — scoring
results = run_scoring(Z, min_h, max_h, 0.68, author_ct_distns, coauthor_graph, df)
# results keys: root, author_index_map, author_meta_distns,
#               doc_scores, link_scores, author_ranking
```

---

## Data Flow

```
arxiv-metadata-oai-snapshot.json
  │
  ▼ load_arxiv_jsonl()         filter by category + year
  │
  ▼ preprocess_text()          tokenise → lemmatise → LDA vocab reduction
  │                            → df["text_processed"], gensim Dictionary
  │
  ▼ disambiguate_authors()     TF-IDF char-trigram cosine + union-find
  │                            → df["author_ids"]
  │
  ▼ create_temporal_chunks()   pd.Grouper(freq="1MS") → list of DataFrames
  │
  ▼ train_ensemble()           per-chunk LDA with exponential smoothing (γ=0.75)
  │                            → topic_vectors (N_total, vocab_size)
  │
  ▼ expand_distributions()     zero-pad chunk-local theta to global topic space
  │                            → expanded_doc_topic_distns {doc_id: (N_total,)}
  │
  ▼ compute_author_barycenters() weighted average over author's documents
  │                            → author_barycenters {author_id: (N_total,)}
  │
  ▼ build_diffusion_graph()    FAISS kNN on topic vectors (Hellinger, k=5)
  │                            bug-fixed: H = √(sq_l2 / 2)
  │
  ▼ diffuse_distributions()    propagate over kNN graph (rate=0.7, 1 iteration)
  │                            single-neighbour guard: use w=1.0 when k=1
  │
  ▼ build_topic_dendrogram()   FAISS kNN (k=100) + Ward linkage on true Hellinger
  │                            → Z linkage matrix (N-1, 4)
  │
  ▼ cut_dendrogram()           fcluster at normalised height 0.68
  │                            → cluster_labels (meta-topic assignments)
  │
  ▼ compute_pairwise_distances() full O(n²) Hellinger matrix for PHATE
  │
  ▼ compute_embedding()        PHATE(knn=5, gamma=0, t=auto, n_components=3)
  │                            → (embedding (N_topics, 3), phate_operator)
  │
  ▼ project_distributions_onto_embedding()   (PHATE only)
  │   cross-Hellinger distances from authors/docs to topics
  │   → barycentric interpolation: mat @ topic_embedding
  │     (weighted average of PHATE topic coords; phate_operator.transform()
  │      is not used — it requires the same data used to fit PHATE)
  │   → author_embedding (N_authors, 3), doc_embedding (N_docs, 3)
  │
  ▼ fast_encode_tree_structure() build HRG binary tree with MLE link probs
  │                            → root TreeNode + author_index_map
  │
  ▼ compute_author_meta_distns() truncate tree at cut; author → meta-topic dist
  │
  ▼ score_interdisciplinarity_docs()  Shannon entropy of doc distributions
  ▼ score_interdisciplinarity_links() HRG link likelihood Σ u⊗v · link_prob_matrix
  ▼ rank_authors_by_interdisciplinarity() sort by entropy descending
  │
  ▼ plot_phate_embedding()     colour by cluster / time / interdisciplinarity
  ▼ generate_meta_topic_wordclouds() per-meta-topic word clouds
  ▼ plot_interdisciplinarity_bars()  top-N most interdisciplinary docs + authors
  ▼ plot_coauthor_network()    edge weight = link likelihood score
```
