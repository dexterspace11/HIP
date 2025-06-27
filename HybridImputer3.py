import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from itertools import product
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# --- Utility Functions ---
def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1e-8
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
    if sum_weights == 0 or np.isnan(sum_weights):
        sum_weights = epsilon
    return weights / sum_weights

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

# --- Streamlit UI ---
st.title("🔮 Hybrid Quantum-Inspired Clustering & Prediction")
mode = st.sidebar.radio("Choose Mode", ["Train Model", "Predict New Data"])

def preprocess_df(df, target_column=None):
    df = df.copy()
    df = df.drop(columns=df.select_dtypes(include=['datetime64']).columns)
    if target_column:
        exclude_cols = [target_column]
    else:
        exclude_cols = []
    # Drop constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=constant_cols)
    # Drop high-cardinality columns (likely identifiers)
    high_card_cols = [col for col in df.columns if df[col].nunique() > 30 and col not in exclude_cols]
    df = df.drop(columns=high_card_cols)
    # One-hot encode categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [col for col in cat_cols if col not in exclude_cols]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # Fill NaNs with median
    for col in df.columns:
        if col not in exclude_cols:
            df[col] = df[col].fillna(df[col].median())
    return df

if mode == "Train Model":
    st.subheader("🧪 Train Clustering Model")
    uploaded_file = st.file_uploader("Upload training Excel or CSV", type=["csv", "xlsx"])
    
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.write("Data Preview", df.head())

        target_column = st.selectbox("Select target column to predict", df.columns)
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)

        target_unique = df[target_column].dropna().unique()
        if df[target_column].dtype == 'O' or len(target_unique) <= 10:
            if len(target_unique) == 2:
                target_type = 'binary'
                threshold = st.slider("Probability threshold for predicting '1'", 0.0, 1.0, 0.5)
            else:
                target_type = 'categorical'
                threshold = None
        else:
            target_type = 'continuous'
            threshold = None

        df_clean = preprocess_df(df, target_column)
        features = [col for col in df_clean.columns if col != target_column]
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df_clean[features])

        param_grid = {
            'alpha': [1.0, 2.0],
            'beta': [0.3, 0.5],
            'gamma': [0.7, 0.9],
            'kappa': [0.05, 0.1]
        }

        best_score = -np.inf
        best_params = None
        best_labels = None

        st.write("Running hyperparameter search...")
        for alpha, beta, gamma, kappa in product(param_grid['alpha'], param_grid['beta'], param_grid['gamma'], param_grid['kappa']):
            labels, _, _ = enhanced_quantum_clustering(data_scaled, n_clusters, alpha, beta, gamma, kappa)
            try:
                score = silhouette_score(data_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_params = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'kappa': kappa}
                    best_labels = labels
            except:
                continue

        st.success(f"Best Silhouette Score: {best_score:.4f}")
        st.json(best_params)

        labels, centroids, weights = enhanced_quantum_clustering(data_scaled, n_clusters, **best_params)
        df_clean['Cluster'] = labels

        if target_type == 'binary':
            cluster_target_map = df_clean.groupby('Cluster')[target_column].mean().to_dict()
            df_clean['Predicted'] = [int(cluster_target_map[label] > threshold) for label in labels]
        elif target_type == 'continuous':
            cluster_target_map = df_clean.groupby('Cluster')[target_column].mean().to_dict()
            df_clean['Predicted'] = [cluster_target_map[label] for label in labels]
        else:  # categorical
            cluster_target_mode = df_clean.groupby('Cluster')[target_column].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            cluster_target_map = cluster_target_mode.to_dict()
            df_clean['Predicted'] = [cluster_target_mode[label] for label in labels]

        model = {
            'centroids': centroids,
            'weights': weights,
            'scaler': scaler,
            'features': features,
            'best_params': best_params,
            'cluster_target_map': cluster_target_map,
            'target_type': target_type,
            'threshold': threshold
        }

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/saved_model.joblib")
        st.success("Model saved to models/saved_model.joblib")

        # Show PCA
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_scaled)
        fig, ax = plt.subplots()
        for cluster in np.unique(labels):
            idx = labels == cluster
            ax.scatter(data_pca[idx, 0], data_pca[idx, 1], label=f'Cluster {cluster}', alpha=0.6)
        ax.set_title("PCA of Clusters")
        ax.legend()
        st.pyplot(fig)

elif mode == "Predict New Data":
    st.subheader("🔮 Predict Using Saved Model")
    uploaded_file = st.file_uploader("Upload new data file", type=["csv", "xlsx"])
    
    if uploaded_file and os.path.exists("models/saved_model.joblib"):
        model = joblib.load("models/saved_model.joblib")
        centroids = model['centroids']
        weights = model['weights']
        scaler = model['scaler']
        features = model['features']
        best_params = model['best_params']
        cluster_target_map = model['cluster_target_map']
        target_type = model['target_type']
        threshold = model.get('threshold', 0.5)

        if uploaded_file.name.endswith(".csv"):
            df_new = pd.read_csv(uploaded_file)
        else:
            df_new = pd.read_excel(uploaded_file)

        df_new = df_new.drop(columns=df_new.select_dtypes(include=['datetime64']).columns)
        # One-hot encode categorical cols in new data if any
        cat_cols = df_new.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            df_new = pd.get_dummies(df_new, columns=cat_cols, drop_first=True)
        for col in features:
            if col not in df_new.columns:
                df_new[col] = 0

        # Fill missing numeric cols with median
        for col in features:
            df_new[col] = df_new[col].fillna(df_new[col].median())

        data_scaled = scaler.transform(df_new[features])
        # Fix: pass only alpha, beta, gamma to assign_clusters
        labels = assign_clusters(
            data_scaled,
            centroids,
            weights,
            best_params['alpha'],
            best_params['beta'],
            best_params['gamma']
        )
        df_new['Cluster'] = labels

        def predict_target(cluster_label):
            val = cluster_target_map.get(cluster_label, np.nan)
            if target_type == 'binary':
                return int(val > threshold)
            return val

        df_new['Predicted'] = df_new['Cluster'].apply(predict_target)
        st.write("Prediction Preview", df_new.head())
        st.download_button("Download Predictions", df_new.to_csv(index=False), "predictions.csv", "text/csv")
    elif not os.path.exists("models/saved_model.joblib"):
        st.warning("Please train and save a model first.")
