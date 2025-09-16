Unsupervised clustering of Netflix titles (movies and TV shows) using TF‑IDF features from textual metadata and one‑hot encoded categorical attributes. The pipeline selects the number of clusters via silhouette analysis, fits K‑Means, and visualizes results with PCA and t‑SNE.

Visuals
PCA projection
![PCA: Netflix Clusters](results/plots/pca_clusters projection (sampled)
![t‑SNE: Netflix Clusters (sample)](results/plots/tsne_clusters

Text features from: director, cast, country, listed_in (genres), description using TF‑IDF.

Categorical features: type, rating, decade derived from release_year via one‑hot encoding.

Model selection: silhouette scores across k to choose cluster count; clustering with K‑Means.

Visualization: PCA for global linear structure; t‑SNE for local neighborhood structure.

Project Structure
src/main.py — load, clean, feature engineering, k‑selection, clustering, summary, CSV export.

src/visualize.py — rebuilds features consistently, merges cluster labels, renders PCA and t‑SNE plots.

data/netflix_titles.csv — dataset CSV (place here after download).

results/cluster_assignments.csv — output with title, type, rating, release_year, cluster.

results/plots/pca_clusters.png, results/plots/tsne_clusters.png — generated figures.

requirements.txt — dependencies.

Setup
Create and activate a virtual environment, then install dependencies.

Windows PowerShell:

python -m venv .venv

.venv\Scripts\Activate.ps1

python -m pip install -U pip

pip install -r requirements.txt

Note: Use recent package versions compatible with the local Python (e.g., pandas ≥ 2.2.3, numpy ≥ 2.1, scikit‑learn ≥ 1.6) to avoid build issues on newer Python releases.

Download the dataset and place it at data/netflix_titles.csv.

Kaggle: Netflix Movies and TV Shows (by Shivam Bansal)

Usage
Run clustering (writes cluster assignments and prints summaries):

python src/main.py

Generate visualizations (saves PCA and t‑SNE plots):

python src/visualize.py

Outputs:

results/cluster_assignments.csv

results/plots/pca_clusters.png

results/plots/tsne_clusters.png

Methodology
TF‑IDF: max_features=5000, English stop‑words, ngram_range=(1,2), min_df=2, max_df=0.8 to balance vocabulary and noise.

One‑hot encoding: type, rating, decade with handle_unknown="ignore".

K selection: silhouette score averaged over a sampled subset for speed; choose k with highest score.

K‑Means: n_init=10 for stability on sparse TF‑IDF features.

Visualization: PCA (2D) on dense array; t‑SNE (2D) on a random sample (e.g., 1,500 rows) to control runtime and memory.

Results (example)
Rows processed: 8,807

Features: 5,030 total (5,000 TF‑IDF + 30 one‑hot)

Selected clusters: k = 2 (highest silhouette)

Interpreted themes:

Cluster 0: TV‑heavy grouping (International TV Shows, TV Dramas/Comedies; strong US/UK presence)

Cluster 1: Movie‑heavy grouping (International Movies, Dramas/Comedies/Documentaries; US/India/UK prominent)

Reproducibility Tips
Ensure the terminal/editor uses the same Python environment as requirements were installed in.

If running in VS Code, select the correct interpreter before executing scripts.

t‑SNE runs on a dense subset for performance; tune sample size, perplexity, and iterations if needed.

Citations
Kaggle Dataset (Netflix Movies and TV Shows): https://www.kaggle.com/datasets/shivamb/netflix-shows

scikit‑learn TfidfVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

scikit‑learn Feature Extraction (text): https://scikit-learn.org/stable/modules/feature_extraction.html

scikit‑learn OneHotEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

scikit‑learn Silhouette Score (API): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

scikit‑learn Silhouette Analysis Example: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

scikit‑learn PCA (API): https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

scikit‑learn t‑SNE (API): https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

scikit‑learn Clustering User Guide: https://scikit-learn.org/stable/modules/clustering.html

VS Code Python (official docs): https://code.visualstudio.com/docs/languages/python

GitHub: Creating a new repository: https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository

GitHub: Quickstart for repositories: https://docs.github.com/en/repositories/creating-and-managing-repositories/quickstart-for-repositories

Package compatibility (Python 3.13) notes:

Pandas (3.13 support issue): https://github.com/pandas-dev/pandas/issues/58734

NumPy (news/PyPI): https://numpy.org/news/ and https://pypi.org/project/numpy/

scikit‑learn (install/PyPI): https://scikit-learn.org/stable/install.html and https://pypi.org/project/scikit-learn/

Matplotlib PyPI: https://pypi.org/project/matplotlib/

License
Add a LICENSE file (e.g., MIT) to specify project terms.

Acknowledgements
Thanks to scikit‑learn maintainers and the Kaggle community for dataset availability.