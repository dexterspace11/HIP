import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import product

# ------------------- Utility Functions ------------------- #

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

# -------------------- Streamlit App --------------------- #

st.title("🧠 Hybrid DNN-EQIC Clustering & Prediction (Quantum-Inspired)")

uploaded_file = st.file_uploader("📁 Upload your Excel or CSV file", type=["xlsx", "csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df.columns = df.columns.astype(str)
    st.write("### 📝 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("⚙️ Train/Test Split")
    manual_split = st.radio("Would you like to manually define the training range?", ["No (automatic split)", "Yes (manual range)"])

    if manual_split == "Yes (manual range)":
        index_column = st.selectbox("Select the index column for range definition", df.columns)
        start = st.number_input("Start row index for training set", min_value=0, max_value=len(df)-1, value=0)
        end = st.number_input("End row index for training set", min_value=0, max_value=len(df)-1, value=len(df)//2)
        train_df = df[(df[index_column] >= df[index_column].iloc[int(start)]) & (df[index_column] <= df[index_column].iloc[int(end)])]
        test_df = df[~df.index.isin(train_df.index)]
    else:
        train_df = df.sample(frac=0.7, random_state=42)
        test_df = df.drop(train_df.index)

    target_column = st.selectbox("🎯 Select the target column to predict", options=df.columns)

    if target_column:
        cluster_count = st.slider("🔢 Number of Clusters", min_value=2, max_value=10, value=3)
        threshold = 0.5
        if df[target_column].nunique() <= 2:
            threshold = st.slider("🎚️ Threshold (for binary targets)", 0.0, 1.0, 0.5)

        df_clean = preprocess_df(train_df, target_column)
        features = [col for col in df_clean.columns if col != target_column]
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df_clean[features])

        st.info("🔍 Tuning hyperparameters...")
        param_grid = {'alpha': [1.0, 2.0], 'beta': [0.3, 0.5], 'gamma': [0.7, 0.9], 'kappa': [0.05, 0.1]}
        labels, best_params, best_score = hyperparameter_search(data_scaled, cluster_count, param_grid)
        st.success(f"✅ Best Silhouette Score: {best_score:.4f}")
        st.json(best_params)

        labels, centroids, weights = enhanced_quantum_clustering(data_scaled, cluster_count, **best_params)
        df_clean["Cluster"] = labels

        cluster_map = df_clean.groupby("Cluster")[target_column].mean().to_dict()
        df_clean["Predicted"] = [cluster_map[c] if df[target_column].nunique() > 2 else int(cluster_map[c] > threshold) for c in labels]

        # Test data
        test_clean = preprocess_df(test_df, target_column)
        test_scaled = scaler.transform(test_clean[features])
        test_labels = assign_clusters(test_scaled, centroids, weights, **{k: best_params[k] for k in ['alpha', 'beta', 'gamma']})
        test_clean["Cluster"] = test_labels
        test_clean["Predicted"] = [cluster_map.get(c, np.nan) if df[target_column].nunique() > 2 else int(cluster_map.get(c, 0) > threshold) for c in test_labels]

        # Combine
        final_df = pd.concat([df_clean, test_clean], axis=0)
        st.subheader("📊 Cluster and Prediction Results")
        st.dataframe(final_df[[target_column, 'Cluster', 'Predicted'] + features].head(10))

        st.subheader("📌 Cluster Summary")
        st.dataframe(final_df.groupby("Cluster")[features + ['Predicted']].agg(['mean', 'std']))

        st.subheader("📉 PCA Visualization")
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(scaler.transform(final_df[features]))
        fig, ax = plt.subplots()
        for i in range(cluster_count):
            ax.scatter(reduced[final_df["Cluster"] == i, 0], reduced[final_df["Cluster"] == i, 1], label=f"Cluster {i}", alpha=0.6)
        ax.set_title("PCA Projection")
        ax.legend()
        st.pyplot(fig)

        # Export
        csv = final_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", data=csv, file_name="hybrid_dnn_eqic_output.csv", mime="text/csv")
