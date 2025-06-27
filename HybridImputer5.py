# hybrid_dnn_eqic_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, accuracy_score, mean_absolute_error
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import product

# -------------------- Utility Functions --------------------

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
    if sum_weights < epsilon:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / sum_weights
    return weights

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

    for it in range(max_iter):
        clusters = [[] for _ in range(n_clusters)]
        for x in data:
            dists = [calculate_distance(x, c, alpha, beta, gamma, weights) for c in centroids]
            clusters[np.argmax(dists)].append(x)

        if it > 0:
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
st.title("🧠 Hybrid DNN-EQIC Clustering and Prediction (Quantum-Inspired)")

uploaded_file = st.file_uploader("📤 Upload your dataset (CSV/XLSX)", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df.columns = df.columns.astype(str)
    st.dataframe(df.head().astype(str))

    st.markdown("### ⚙️ Train/Test Configuration")
    auto_split = st.radio("Auto-split train/test?", ["Yes", "No"], index=0)

    if auto_split == "No":
        split_column = st.selectbox("Reference column for split", df.columns)
        min_val, max_val = int(df[split_column].min()), int(df[split_column].max())
        train_start = st.number_input("Train Start", min_value=min_val, max_value=max_val, value=min_val)
        train_end = st.number_input("Train End", min_value=min_val, max_value=max_val, value=min_val+10)
        train_df = df[(df[split_column] >= train_start) & (df[split_column] <= train_end)]
        test_df = df[~((df[split_column] >= train_start) & (df[split_column] <= train_end))]

    else:
        train_ratio = st.slider("Train ratio", 0.1, 0.9, 0.7)
        split_idx = int(len(df) * train_ratio)
        train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    st.markdown("### 🎯 Prediction Setup")
    target_column = st.selectbox("Select target column", df.columns)
    threshold = st.slider("Binary threshold (if applicable)", 0.0, 1.0, 0.5)
    n_clusters = st.slider("Number of clusters", 2, 10, 3)

    if st.button("🚀 Start Analysis"):
        train_clean = preprocess_df(train_df, target_column)
        test_clean = preprocess_df(test_df, target_column)
        test_clean = test_clean.reindex(columns=train_clean.columns, fill_value=0)

        features = [col for col in train_clean.columns if col != target_column]
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_clean[features])
        test_scaled = scaler.transform(test_clean[features])

        st.info("🔍 Tuning hyperparameters...")
        param_grid = {'alpha': [1.0, 2.0], 'beta': [0.3, 0.5], 'gamma': [0.7, 0.9], 'kappa': [0.05, 0.1]}
        _, best_params, score = hyperparameter_search(train_scaled, n_clusters, param_grid)
        st.success(f"✅ Best Silhouette Score: {score:.4f}")
        st.json(best_params)

        labels_train, centroids, weights = enhanced_quantum_clustering(train_scaled, n_clusters, **best_params)
        train_clean['Cluster'] = labels_train
        labels_test = assign_clusters(test_scaled, centroids, weights,
                                      alpha=best_params['alpha'], beta=best_params['beta'], gamma=best_params['gamma'])
        test_clean['Cluster'] = labels_test

        # Predict target
        target_type = 'binary' if train_clean[target_column].nunique() == 2 else ('categorical' if train_clean[target_column].dtype == 'O' else 'continuous')

        def predict_from_cluster(df):
            mapping = df.groupby('Cluster')[target_column].mean().to_dict() if target_type in ['binary', 'continuous'] else df.groupby('Cluster')[target_column].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
            if target_type == 'binary':
                return [int(mapping[l] > threshold) for l in df['Cluster']]
            return [mapping.get(l, np.nan) for l in df['Cluster']]

        train_clean['Predicted'] = predict_from_cluster(train_clean)
        test_clean['Predicted'] = predict_from_cluster(test_clean)

        # Metrics
        if target_type == 'binary':
            acc = accuracy_score(train_clean[target_column], train_clean['Predicted'])
            st.metric("Train Accuracy", f"{acc:.2%}")
        elif target_type == 'continuous':
            mae = mean_absolute_error(train_clean[target_column], train_clean['Predicted'])
            st.metric("Train MAE", f"{mae:.4f}")

        # PCA visualization
        st.markdown("### 🧬 PCA Cluster Projection")
        combined_scaled = np.vstack([train_scaled, test_scaled])
        combined_labels = np.concatenate([labels_train, labels_test])
        pca = PCA(n_components=2)
        proj = pca.fit_transform(combined_scaled)

        fig, ax = plt.subplots()
        for c in np.unique(combined_labels):
            ax.scatter(proj[combined_labels==c, 0], proj[combined_labels==c, 1], label=f"Cluster {c}")
        ax.set_title("Clusters (PCA Projection)")
        ax.legend()
        st.pyplot(fig)

        # Centroids
        st.markdown("### 🔍 Centroid Feature Vectors")
        st.dataframe(pd.DataFrame(centroids, columns=features).astype(str))

        # Downloadable output
        result_df = pd.concat([
            train_df.assign(Cluster=labels_train, Predicted=train_clean['Predicted']),
            test_df.assign(Cluster=labels_test, Predicted=test_clean['Predicted'])
        ]).sort_index()

        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, file_name="hybrid_dnn_eqic_output.csv")
