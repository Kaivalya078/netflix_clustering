Netflix Show Clustering
Unsupervised clustering of Netflix titles using TF‑IDF features from textual metadata and one‑hot encoded categorical attributes. The pipeline selects the number of clusters via silhouette analysis, fits K‑Means, and visualizes results with PCA and t‑SNE.

Overview
Text features from director, cast, country, listed_in (genres), description via TF‑IDF.

Categorical features: type, rating, decade (derived from release_year) via one‑hot encoding.

Model selection: silhouette scores scanned across k to select cluster count; K‑Means for clustering.

Visualization: PCA (linear) and t‑SNE (non‑linear) 2D embeddings.

Project Structure
src/main.py — load, clean, feature engineering, k‑selection, clustering, summary, CSV export.

src/visualize.py — rebuild features, merge cluster labels, render PCA and t‑SNE plots.

data/netflix_titles.csv — dataset CSV (download and place here).

results/cluster_assignments.csv — output with title, type, rating, release_year, cluster.

results/plots/pca_clusters.png, results/plots/tsne_clusters.png — generated figures.

Setup
Windows PowerShell:

python -m venv .venv

.venv\Scripts\Activate.ps1

python -m pip install -U pip

pip install -r requirements.txt

Dataset:

Download from Kaggle (Netflix Movies and TV Shows by Shivam Bansal) and save as data/netflix_titles.csv.

Usage
Run clustering (writes cluster assignments and prints summary):

python src/main.py

Generate visualizations:

python src/visualize.py

Outputs:

results/cluster_assignments.csv

results/plots/pca_clusters.png

results/plots/tsne_clusters.png

Methodology
TF‑IDF: max_features=5000, English stop‑words, ngram_range=(1,2), min_df=2, max_df=0.8.

One‑hot encoding: type, rating, decade with handle_unknown="ignore".

K selection: silhouette score on a sampled subset; choose k with highest score.

K‑Means: n_init=10 on sparse features.

Visualization: PCA on dense array; t‑SNE on a random sample for performance.

Results (example)
Rows processed: 8,807

Features: 5,030 total (5,000 TF‑IDF + 30 one‑hot)

Selected clusters: k = 2 (highest silhouette)

Themes:

Cluster 0: TV‑heavy grouping (International TV Shows, TV Dramas/Comedies)

Cluster 1: Movie‑heavy grouping (International Movies, Dramas/Comedies/Documentaries)

Citations
Kaggle Dataset: https://www.kaggle.com/datasets/shivamb/netflix-shows

scikit‑learn TfidfVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

scikit‑learn Feature Extraction (text): https://scikit-learn.org/stable/modules/feature_extraction.html

scikit‑learn OneHotEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

scikit‑learn Silhouette Score (API): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

Silhouette Analysis Example: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

scikit‑learn PCA (API): https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

scikit‑learn t‑SNE (API): https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

scikit‑learn Clustering Guide: https://scikit-learn.org/stable/modules/clustering.html

VS Code Python (docs): https://code.visualstudio.com/docs/languages/python


Acknowledgements
Thanks to scikit‑learn maintainers and the Kaggle community.
