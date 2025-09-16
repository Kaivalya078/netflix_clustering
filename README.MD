# Netflix Show Clustering

Unsupervised clustering of Netflix titles using TF-IDF features from textual metadata and one-hot encoded categorical attributes. The pipeline selects optimal cluster count via silhouette analysis, fits K-Means, and provides PCA and t-SNE visualizations for interpretability.

## Features

- **End-to-end clustering pipeline**: Load, clean, engineer features, select k, cluster, and export results with a single command.
- **Robust feature engineering**: TF-IDF on combined text (director, cast, country, genres, description) + one-hot encoding for categorical data.
- **Automated model selection**: Silhouette analysis across k=[2,10] to choose optimal cluster count.
- **Rich visualizations**: PCA for linear projection and t-SNE for non-linear neighborhood structure.
- **Clean output**: CSV assignments and high-DPI PNG plots saved to results/ folder.

## Project Structure

```
netflix_clustering/
├─ data/
│ └─ netflix_titles.csv # Dataset (download from Kaggle)
├─ results/
│ ├─ cluster_assignments.csv # Output with title, type, rating, release_year, cluster
│ └─ plots/ # Generated visualizations
│ ├─ pca_clusters.png # PCA 2D projection
│ └─ tsne_clusters.png # t-SNE 2D projection
├─ src/
│ ├─ main.py # End-to-end pipeline
│ └─ visualize.py # Generate PCA and t-SNE plots
├─ .gitignore
├─ README.md
└─ requirements.txt

## Prerequisites

- Python 3.8+ with packages from requirements.txt
- Netflix Movies and TV Shows dataset from Kaggle

## Installation

Create a virtual environment and install dependencies:

**macOS/Linux:**
```
python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

**Windows (PowerShell):**
```
python -m venv .venv

..venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt

## Dataset Setup

Download the Netflix dataset and place it in the data/ folder:

1. Visit [Kaggle: Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows)
2. Download `netflix_titles.csv`
3. Save as `data/netflix_titles.csv`

## How to Run

**Step 1: Run clustering pipeline**
```
python src/main.py

**Step 2: Generate visualizations**
```
python src/visualize.py

**What it produces:**
- `results/cluster_assignments.csv` - Clustering results with metadata
- `results/plots/pca_clusters.png` - PCA 2D projection
- `results/plots/tsne_clusters.png` - t-SNE 2D visualization

## Methodology

- **Text Features**: TF-IDF with 5,000 max features, English stop-words, bigrams, min_df=2, max_df=0.8
- **Categorical Features**: One-hot encoding for type, rating, and release decade
- **Model Selection**: Silhouette score optimization across k=[2,10] on sampled subset
- **Clustering**: K-Means with multiple initializations (n_init=10) for stability
- **Visualization**: PCA on full dataset; t-SNE on 1,500-sample subset for performance

## How It Works

- `src/main.py` orchestrates the full pipeline: data loading, cleaning, TF-IDF vectorization, one-hot encoding, feature stacking, silhouette-based k-selection, K-Means clustering, and results export.
- `src/visualize.py` rebuilds the feature matrix consistently, merges cluster assignments, and generates PCA/t-SNE plots with proper sampling for performance.
- Results are saved to `results/` with both tabular assignments (CSV) and visual summaries (PNG).

## Customization

- **Adjust cluster range**: Modify `--kmin` and `--kmax` arguments in main.py
- **Change feature limits**: Edit TF-IDF `max_features` parameter for vocabulary size
- **Sampling for visualization**: Adjust `sample_cap` in t-SNE generation for speed/quality tradeoffs
- **Additional plots**: Extend visualize.py with custom matplotlib/seaborn charts

## Reproducibility

- Fixed random seeds (42) ensure consistent results across runs
- Compatible with Python 3.8-3.13 and recent scikit-learn/pandas versions
- All dependencies pinned in requirements.txt for environment consistency

## Acknowledgments

- **Dataset**: Netflix Movies and TV Shows by Shivam Bansal (Kaggle)
- **Built with**: scikit-learn, pandas, matplotlib, seaborn, scipy, numpy


## Citations

- [Kaggle Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [TF-IDF Guide](https://scikit-learn.org/stable/modules/feature_extraction.html)
- [Silhouette Analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
