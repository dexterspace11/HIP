# hybrid_eqic_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import product

# ---------------------------- Utility Functions -----------------------------

def preprocess_df(df, target_column):
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    df = df.drop(columns=df.select_dtypes(include=['datetime64']).columns)
    df = df.loc[:, df.nunique() > 1]
    high_card_cols = [col for col in df.columns if df[col].nunique() > 30 and col != target_column]
    df = df.drop(columns=high_card_cols)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    cat_cols = [c for c in cat_cols if c != target_column]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    for col in df.columns:
        if col != target_column:
            df[col] = df[col].fillna(df[col].median())
    return df

def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1e-8
    return (data - mean) / std

def calculate_distance(point, centroid, alpha, beta, gamma, weights):
    weighted_diff = weights * np.abs(point - centroid)
    dist = np.sqrt(np.sum(weighted_diff ** 2))
    exp_term = np.exp(-alpha * dist)
    inv_term = beta / (1 + gamma * dist)
    return exp_term + inv_term

def centroid_interaction(c1, c2, kappa):
    dist = np.linalg.norm(c1 - c2)
    return np.exp(-kappa * dist)

def update_dimension_weights(clusters, data):
    weights = np.ones(data.shape[1])
    epsilon = 1e-8
    for cluster in clusters:
        if cluster:
            cluster_data = np.array(cluster)
            var = np.var(cluster_data, axis=0)
            mean_var = np.mean(var) if np.mean(var) != 0 else epsilon
            weights *= var / mean_var
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    return weights / np.sum(weights)

def update_centroids(clusters, centroids, gamma, kappa):
    new_centroids = []
    for i, cluster in enumerate(clusters):
        if cluster:
            mean = np.mean(cluster, axis=0)
            interaction = sum(
                centroid_interaction(centroids[i], centroids[j], kappa) * centroids[j]
                for j in range(len(centroids)) if j != i
            ) / (len(centroids) - 1)
            new_c = gamma * mean + (1 - gamma) * (centroids[i] + interaction)
            new_centroids.append(new_c)
        else:
            new_centroids.append(centroids[i])
    return np.array(new_centroids)

def assign_clusters(data, centroids, weights, alpha, beta, gamma):
    labels = []
    for point in data:
        dists = [calculate_distance(point, c, alpha, beta, gamma, weights) for c in centroids]
        labels.append(np.argmax(dists))
    return np.array(labels)

def enhanced_quantum_clustering(data, n_clusters=2, alpha=2.0, beta=0.5, gamma=0.9, kappa=0.1, tol=1e-4, max_iter=100):
    data = normalize_data(data)
    idx = np.random.choice(len(data), n_clusters, replace=False)
    centroids = data[idx]
    weights = np.ones(data.shape[1])

    for _ in range(max_iter):
        clusters = [[] for _ in range(n_clusters)]
        for x in data:
            dists = [calculate_distance(x, c, alpha, beta, gamma, weights) for c in centroids]
            clusters[np.argmax(dists)].append(x)

        weights = update_dimension_weights(clusters, data)
        new_centroids = update_centroids(clusters, centroids, gamma, kappa)

        if np.max(np.abs(new_centroids - centroids)) < tol:
            break
        centroids = new_centroids

    labels = assign_clusters(data, centroids, weights, alpha, beta, gamma)
    return labels, centroids, weights

def hyperparameter_search(data, n_clusters, param_grid):
    best_score = -np.inf
    best_params = None
    best_labels = None
    for alpha, beta, gamma, kappa in product(param_grid['alpha'], param_grid['beta'], param_grid['gamma'], param_grid['kappa']):
        labels, _, _ = enhanced_quantum_clustering(
            data, n_clusters=n_clusters, alpha=alpha, beta=beta, gamma=gamma, kappa=kappa
        )
        try:
            score = silhouette_score(data, labels)
        except:
            score = -1
        if score > best_score:
            best_score = score
            best_params = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'kappa': kappa}
            best_labels = labels
    return best_labels, best_params, best_score

# ----------------------------- Streamlit UI --------------------------------

st.set_page_config(page_title="Hybrid DNN-EQIC Predictor", layout="wide")
st.title("🧠 Hybrid DNN-EQIC Quantum-Inspired Clustering & Prediction")
uploaded_file = st.file_uploader("Upload Dataset (Excel/CSV)", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

    col_id = st.text_input("Optional: Column for Range-Based Split (e.g., 'id')", value="")
    if col_id and col_id in df.columns:
        range_start = st.number_input("Start of Training Range", value=0)
        range_end = st.number_input("End of Training Range", value=int(len(df)*0.7))
        train_df = df[df[col_id].between(range_start, range_end)].copy()
        test_df = df[~df[col_id].between(range_start, range_end)].copy()
    else:
        split_ratio = st.slider("Train/Test Split Ratio", 0.1, 0.9, 0.7)
        train_df = df.sample(frac=split_ratio, random_state=42)
        test_df = df.drop(train_df.index)

    target_col = st.selectbox("Select Target Variable to Predict", df.columns)
    threshold = st.slider("Threshold (for binary targets)", 0.0, 1.0, 0.5)
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)

    df_clean = preprocess_df(df, target_col)
    features = [f for f in df_clean.columns if f != target_col]
    scaled = MinMaxScaler().fit_transform(df_clean[features])

    st.info("Running Quantum-Inspired Clustering with Memory...")
    param_grid = {'alpha': [1.0, 2.0], 'beta': [0.3, 0.5], 'gamma': [0.7, 0.9], 'kappa': [0.05, 0.1]}
    labels, best_params, score = hyperparameter_search(scaled, n_clusters, param_grid)
    st.success(f"Best Silhouette Score: {score:.4f}")
    st.json(best_params)

    labels, centroids, weights = enhanced_quantum_clustering(scaled, n_clusters, **best_params)
    df_clean['Cluster'] = labels

    # Mapping Clusters to Predicted Target
    t = df_clean[target_col]
    if t.nunique() <= 2:
        cluster_map = df_clean.groupby('Cluster')[target_col].mean().to_dict()
        df_clean['Predicted'] = df_clean['Cluster'].map(lambda x: int(cluster_map[x] > threshold))
    elif t.dtype == float or t.dtype == int:
        cluster_map = df_clean.groupby('Cluster')[target_col].mean().to_dict()
        df_clean['Predicted'] = df_clean['Cluster'].map(cluster_map)
    else:
        cluster_map = df_clean.groupby('Cluster')[target_col].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan).to_dict()
        df_clean['Predicted'] = df_clean['Cluster'].map(cluster_map)

    # Cluster Statistics
    st.subheader("🔍 Cluster Summary")
    st.write(df_clean.groupby('Cluster')[features + ['Predicted']].agg(['mean', 'std']))
    st.write("Cluster Sizes:", df_clean['Cluster'].value_counts())

    # Centroid Insights
    st.subheader("🧠 Centroid Analysis")
    centroid_df = pd.DataFrame(centroids, columns=features)
    st.dataframe(centroid_df)

    # PCA Visualization
    try:
        st.subheader("📊 PCA Cluster Visualization")
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(scaled)
        fig, ax = plt.subplots()
        for i in range(n_clusters):
            idx = df_clean['Cluster'] == i
            ax.scatter(reduced[idx, 0], reduced[idx, 1], label=f"Cluster {i}", alpha=0.6)
        ax.legend()
        ax.set_title("PCA of Clusters")
        st.pyplot(fig)
    except:
        st.warning("PCA could not be plotted.")

    # Final Output
    st.subheader("📥 Download Results")
    df_final = pd.concat([df.reset_index(drop=True), df_clean[['Cluster', 'Predicted']]], axis=1)
    st.dataframe(df_final.head(10))
    st.download_button("Download CSV", data=df_final.to_csv(index=False), file_name="final_output.csv", mime="text/csv")
