import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import product

# ------------- Utility Functions -------------------

def preprocess_df(df, target_column):
    df = df.copy()

    # Force object-type columns to strings to avoid pyarrow issues
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    # Drop datetime columns
    df = df.drop(columns=df.select_dtypes(include=['datetime64']).columns)

    # Drop constant columns
    df = df.loc[:, df.nunique() > 1]

    # Drop high-cardinality columns (likely IDs), except target
    high_card_cols = [col for col in df.columns if df[col].nunique() > 30 and col != target_column]
    df = df.drop(columns=high_card_cols)

    # One-hot encode categorical (except target)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    cat_cols = [c for c in cat_cols if c != target_column]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Fill NaNs with median (exclude target)
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

# ------------- Streamlit UI -------------------

st.title("🔮 Hybrid Clustering Predictor (Quantum-Inspired)")
uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = df.columns.astype(str)
    st.write("Dataset Preview:")
    st.dataframe(df.astype(str).head())

    target_column = st.selectbox("Select Target Column to Predict", options=df.columns)

    if target_column:
        df_clean = preprocess_df(df, target_column)
        features = [col for col in df_clean.columns if col != target_column]

        # Determine type of target
        target_unique = df[target_column].dropna().unique()
        if df[target_column].dtype == 'O' or len(target_unique) <= 10:
            target_type = 'binary' if len(target_unique) == 2 else 'categorical'
        else:
            target_type = 'continuous'
        st.write(f"Detected target type: `{target_type}`")

        threshold = 0.5
        if target_type == 'binary':
            threshold = st.slider("Select Threshold for Binary Prediction", 0.0, 1.0, 0.5)

        # Scaling
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df_clean[features])

        # Hyperparameter tuning
        param_grid = {
            'alpha': [1.0, 2.0],
            'beta': [0.3, 0.5],
            'gamma': [0.7, 0.9],
            'kappa': [0.05, 0.1]
        }

        st.info("⏳ Searching for best hyperparameters...")
        labels, best_params, best_score = hyperparameter_search(data_scaled, n_clusters=st.slider("Number of Clusters", 2, 6, 3), param_grid=param_grid)
        st.success(f"✅ Best Silhouette Score: {best_score:.4f}")
        st.json(best_params)

        # Final clustering
        labels, centroids, weights = enhanced_quantum_clustering(data_scaled, n_clusters=len(set(labels)), **best_params)
        df_clean["Cluster"] = labels

        # Cluster-to-target mapping
        if target_type == 'binary':
            cluster_map = df_clean.groupby('Cluster')[target_column].mean().to_dict()
            df_clean['Predicted'] = [int(cluster_map[l] > threshold) for l in labels]
        elif target_type == 'continuous':
            cluster_map = df_clean.groupby('Cluster')[target_column].mean().to_dict()
            df_clean['Predicted'] = [cluster_map[l] for l in labels]
        else:
            cluster_map = df_clean.groupby('Cluster')[target_column].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
            df_clean['Predicted'] = [cluster_map[l] for l in labels]

        st.subheader("📊 Cluster Analysis")
        st.write("Cluster Mean/Std:")
        st.dataframe(df_clean.groupby('Cluster')[features + ['Predicted']].agg(['mean', 'std']))

        st.write("Cluster Counts:")
        st.dataframe(df_clean['Cluster'].value_counts().rename("Count"))

        # PCA plot
        try:
            st.subheader("🌀 PCA Cluster Plot")
            pca = PCA(n_components=2)
            data_pca = pca.fit_transform(data_scaled)
            fig, ax = plt.subplots(figsize=(8, 6))
            for cluster in np.unique(labels):
                idx = labels == cluster
                ax.scatter(data_pca[idx, 0], data_pca[idx, 1], label=f"Cluster {cluster}", alpha=0.6)
            ax.set_title("PCA Projection of Clusters")
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"PCA plot not available: {e}")

        st.subheader("🔍 Final Output")
        st.dataframe(df_clean[[target_column, 'Cluster', 'Predicted'] + features].astype(str).head(10))

        # Downloadable
        csv = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", data=csv, file_name="cluster_predictions.csv", mime="text/csv")

