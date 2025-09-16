import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_and_clean(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path)

    df['director'] = df['director'].fillna('Unknown Director')
    df['cast'] = df['cast'].fillna('Unknown Cast')
    df['country'] = df['country'].fillna('Unknown Country')
    df['rating'] = df['rating'].fillna('Not Rated')
    df['description'] = df['description'].fillna('No Description')
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['decade'] = (df['release_year'] // 10) * 10

    df['combined_text'] = (
        df['director'].astype(str) + ' ' +
        df['cast'].astype(str) + ' ' +
        df['country'].astype(str) + ' ' +
        df['listed_in'].astype(str) + ' ' +
        df['description'].astype(str)
    )
    return df

def build_features(df: pd.DataFrame):
    print("Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    X_text = tfidf.fit_transform(df['combined_text'])
    print(f"TF-IDF shape: {X_text.shape[0]} rows x {X_text.shape[1]} cols")

    print("One-hot encoding categorical features...")
    cats = df[['type', 'rating', 'decade']].astype(str)
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)
    X_cat = ohe.fit_transform(cats)
    if not issparse(X_cat):
        X_cat = csr_matrix(X_cat)
    print(f"OHE shape: {X_cat.shape[0]} rows x {X_cat.shape[1]} cols")

    if X_text.shape[0] != X_cat.shape[0]:
        raise ValueError(f"Row mismatch: TF-IDF={X_text.shape[0]} vs OHE={X_cat.shape[0]}")

    print("Stacking features...")
    X = hstack([X_text, X_cat], format='csr')
    print(f"Final feature matrix: {X.shape[0]} rows x {X.shape[1]} cols (sparse CSR)")
    return X, tfidf, ohe

def select_k_by_silhouette(X, k_min=2, k_max=10, sample_cap=1500, random_state=42):
    ks = list(range(k_min, k_max + 1))
    n = X.shape[0]
    sample_size = min(sample_cap, n)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, sample_size, replace=False)

    sils = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X[idx], labels[idx])
        sils.append(sil)

    best_k = ks[int(np.argmax(sils))]
    return best_k, ks, sils

def summarize_clusters(df: pd.DataFrame, label_col: str = 'cluster', top_genres=5, top_countries=3) -> pd.DataFrame:
    rows = []
    for c in sorted(df[label_col].unique()):
        sub = df[df[label_col] == c]
        genres = sub['listed_in'].str.split(',').explode().str.strip().value_counts().head(top_genres).index.tolist()
        countries = sub['country'].str.split(',').explode().str.strip().value_counts().head(top_countries).index.tolist()
        rows.append({
            'cluster': c,
            'size': int(len(sub)),
            'top_genres': genres,
            'top_countries': countries
        })
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Netflix Show Clustering")
    parser.add_argument("--data", type=str, default=os.path.join("data", "netflix_titles.csv"))
    parser.add_argument("--out", type=str, default=os.path.join("results", "cluster_assignments.csv"))
    parser.add_argument("--kmin", type=int, default=2)
    parser.add_argument("--kmax", type=int, default=10)
    parser.add_argument("--sample_cap", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading and cleaning data...")
    df = load_and_clean(args.data)
    print(f"Loaded {len(df)} rows")

    print("Building features (TF-IDF + One-Hot)...")
    X, tfidf, ohe = build_features(df)

    print(f"Selecting k via silhouette in [{args.kmin}, {args.kmax}]...")
    optimal_k, ks, sils = select_k_by_silhouette(
        X, k_min=args.kmin, k_max=args.kmax, sample_cap=args.sample_cap, random_state=args.seed
    )
    print("k values:", ks)
    print("silhouette scores:", [round(s, 4) for s in sils])
    print("Optimal k:", optimal_k)

    print("Fitting final KMeans...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=args.seed, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)

    print("Summarizing clusters...")
    summary_df = summarize_clusters(df, label_col='cluster')
    print(summary_df.to_string(index=False))

    out_cols = ['show_id', 'title', 'type', 'rating', 'release_year', 'cluster']
    existing = [c for c in out_cols if c in df.columns]
    df[existing].to_csv(args.out, index=False)
    print(f"Saved cluster assignments to: {args.out}")

if __name__ == "__main__":
    main()
