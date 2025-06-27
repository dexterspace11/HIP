# hybrid_dnn_eqic_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, accuracy_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from itertools import product
import warnings
warnings.filterwarnings("ignore")

# -------------------- Utility Functions --------------------
def preprocess_df(df, target_column=None):
    df = df.copy()
    df = df.drop(columns=df.select_dtypes(include=['datetime64', 'datetime']).columns, errors='ignore')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    df = df.loc[:, df.nunique(dropna=False) > 1]

    target = df[target_column] if target_column and target_column in df.columns else None
    if target_column and target_column in df.columns:
        df = df.drop(columns=[target_column])

    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df = df.fillna(df.median(numeric_only=True))

    if target_column and target is not None:
        df[target_column] = target

    usable_cols = [col for col in df.columns if col != target_column]
    if not usable_cols:
        fallback = ['open', 'high', 'low', 'close', 'volume', 'VWAP']
        fallback_cols = [col for col in fallback if col in df.columns]
        if not fallback_cols:
            raise ValueError("No features available after preprocessing. Please check your dataset.")
        return df[fallback_cols + ([target_column] if target_column else [])]

    return df

def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std = np.where(std == 0, 1e-8, std)
    return (data - mean) / std

def calculate_distance(point, centroid, alpha, beta, gamma, weights):
    weighted_diff = weights * np.abs(point - centroid)
    dist = np.sqrt(np.sum(weighted_diff**2))
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
    sum_weights = np.sum(weights)
    return weights / sum_weights if sum_weights >= epsilon else np.ones_like(weights) / len(weights)

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

        if _ > 0:
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

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Hybrid DNN-EQIC Clustering", layout="wide")
st.title("🧐 Hybrid DNN-EQIC Clustering + Adaptive Forecasting")

uploaded_file = st.file_uploader("📄 Upload your training dataset (CSV/XLSX)", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df.columns = df.columns.astype(str)
    st.dataframe(df.head())

    st.markdown("### 🎯 Clustering Setup")
    target_column = st.selectbox("Select target column (optional)", ["None"] + list(df.columns))
    target_column = None if target_column == "None" else target_column
    n_clusters = st.slider("Number of clusters", 2, 10, 3)

    if st.button("🚀 Start Clustering"):
        try:
            df_clean = preprocess_df(df, target_column)
        except ValueError as e:
            st.error(f"🚫 {str(e)}")
            st.stop()

        features = [col for col in df_clean.columns if col != target_column]
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(df_clean[features])

        st.info("🔍 Hyperparameter tuning in progress...")
        param_grid = {'alpha': [1.0, 2.0], 'beta': [0.3, 0.5], 'gamma': [0.7, 0.9], 'kappa': [0.05, 0.1]}
        _, best_params, sil_score = hyperparameter_search(X_scaled, n_clusters, param_grid)
        st.success(f"✅ Silhouette Score: {sil_score:.4f}")
        st.json(best_params)

        labels, centroids, weights = enhanced_quantum_clustering(X_scaled, n_clusters, **best_params)
        df_clean['Cluster'] = labels

        st.markdown("### 📊 Clustering Metrics")
        st.write(f"Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, labels):.2f}")
        st.write(f"Davies-Bouldin Score: {davies_bouldin_score(X_scaled, labels):.2f}")

        if target_column:
            if pd.api.types.is_numeric_dtype(df_clean[target_column]):
                cluster_map = df_clean.groupby('Cluster')[target_column].mean().to_dict()
                intra_error = df_clean.groupby('Cluster').apply(lambda x: (x[target_column] - x[target_column].mean()).mean()).to_dict()
            else:
                cluster_map = df_clean.groupby('Cluster')[target_column].agg(lambda x: x.mode().iloc[0]).to_dict()
                intra_error = {k: 0 for k in cluster_map.keys()}

            df_clean['Predicted'] = df_clean['Cluster'].map(cluster_map)
            df_clean['Predicted'] += df_clean['Cluster'].map(intra_error)

            if pd.api.types.is_numeric_dtype(df_clean[target_column]):
                st.metric("Cluster Forecast MAE", f"{np.mean(np.abs(df_clean[target_column] - df_clean['Predicted'])):.4f}")
            else:
                acc = accuracy_score(df_clean[target_column], df_clean['Predicted'])
                st.metric("Cluster Match Accuracy", f"{acc:.2%}")

        st.markdown("### 📌 PCA / t-SNE Visualization")
        tab1, tab2 = st.tabs(["PCA", "t-SNE"])

        with tab1:
            pca = PCA(n_components=2)
            proj = pca.fit_transform(X_scaled)
            fig, ax = plt.subplots()
            for c in np.unique(labels):
                ax.scatter(proj[labels == c, 0], proj[labels == c, 1], label=f"Cluster {c}", alpha=0.6)
            ax.set_title("PCA Cluster Projection")
            ax.legend()
            st.pyplot(fig)

        with tab2:
            tsne = TSNE(n_components=2, perplexity=30, n_iter=500)
            tsne_proj = tsne.fit_transform(X_scaled)
            fig, ax = plt.subplots()
            for c in np.unique(labels):
                ax.scatter(tsne_proj[labels == c, 0], tsne_proj[labels == c, 1], label=f"Cluster {c}", alpha=0.6)
            ax.set_title("t-SNE Cluster Projection")
            ax.legend()
            st.pyplot(fig)

        st.markdown("### 🔍 Cluster Centroids")
        st.dataframe(pd.DataFrame(centroids, columns=features))

        csv = df_clean.to_csv(index=False).encode('utf-8', errors='ignore')
        st.download_button("📅 Download Clustered Data", csv, file_name="clustered_output.csv")

        # ------------- Forecasting New Data --------------------
        st.markdown("---")
        st.markdown("### 📈 Predict on New Data")
        new_file = st.file_uploader("📄 Upload new dataset for prediction (same features)", type=["csv", "xlsx"], key="new")

        if new_file:
            new_df = pd.read_csv(new_file) if new_file.name.endswith(".csv") else pd.read_excel(new_file)
            new_df.columns = new_df.columns.astype(str)
            try:
                new_df_clean = preprocess_df(new_df)
                new_scaled = scaler.transform(new_df_clean[features])
                new_labels = assign_clusters(new_scaled, centroids, weights, **best_params)
                new_df_clean['Cluster'] = new_labels
                new_df_clean['Predicted'] = new_df_clean['Cluster'].map(cluster_map)
                new_df_clean['Predicted'] += new_df_clean['Cluster'].map(intra_error)
                st.dataframe(new_df_clean.head())

                forecast_csv = new_df_clean.to_csv(index=False).encode('utf-8', errors='ignore')
                st.download_button("📅 Download Predictions", forecast_csv, file_name="forecast_output.csv")
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")