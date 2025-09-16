import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import hstack
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = os.path.join("data", "netflix_titles.csv")
ASSIGN_PATH = os.path.join("results", "cluster_assignments.csv")
PLOT_DIR = os.path.join("results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_PATH)
assign = pd.read_csv(ASSIGN_PATH)

# Clean + engineer same as training
df['director'] = df['director'].fillna('Unknown Director')
df['cast'] = df['cast'].fillna('Unknown Cast')
df['country'] = df['country'].fillna('Unknown Country')
df['rating'] = df['rating'].fillna('Not Rated')
df['description'] = df['description'].fillna('No Description')
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['decade'] = (df['release_year'] // 10) * 10

df['combined_text'] = (
    df['director'].astype(str) + ' ' +
    df['cast'].astype(str) + ' ' +
    df['country'].astype(str) + ' ' +
    df['listed_in'].astype(str) + ' ' +
    df['description'].astype(str)
)

# Vectorize text
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)
X_text = tfidf.fit_transform(df['combined_text'])

# One-Hot categorical
try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
except TypeError:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)

X_cat = ohe.fit_transform(df[['type', 'rating', 'decade']].astype(str))
X = hstack([X_text, X_cat])

# Merge cluster labels by title
df_vis = df[['title']].copy()
df_vis = df_vis.merge(assign[['title', 'cluster']], on='title', how='left')
labels = df_vis['cluster'].values

# PCA visualization (dense projection)
print("PCA projection...")
pca = PCA(n_components=2, random_state=42)
Xp = pca.fit_transform(X.toarray())

plt.figure(figsize=(7, 6))
sns.scatterplot(x=Xp[:, 0], y=Xp[:, 1], hue=labels, palette='tab10', s=10, linewidth=0)
plt.title('PCA: Netflix Clusters')
plt.legend(title='cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "pca_clusters.png"), dpi=150)
plt.close()

# t-SNE visualization (sampled for speed/memory)
print("t-SNE projection (sampling for speed)...")
n = X.shape[0]                      # number of rows
sample = min(1500, n)               # compare ints to avoid tuple/int errors
rng = np.random.default_rng(42)
idx = rng.choice(n, sample, replace=False)

# Use max_iter instead of n_iter for newer scikit-learn versions
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
    learning_rate='auto',
    max_iter=1000
)
Xt = tsne.fit_transform(X[idx].toarray())

plt.figure(figsize=(7, 6))
sns.scatterplot(x=Xt[:, 0], y=Xt[:, 1], hue=labels[idx], palette='tab10', s=12, linewidth=0)
plt.title('t-SNE: Netflix Clusters (sample)')
plt.legend(title='cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "tsne_clusters.png"), dpi=150)
plt.close()

print("Saved plots: results/plots/pca_clusters.png, results/plots/tsne_clusters.png")
