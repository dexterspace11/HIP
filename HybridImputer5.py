import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import product

# ----- Hybrid DNN-EQIC Core Functions -----

def preprocess_df(df, target_column):
    df = df.copy()

    # Convert all object columns to string for PyArrow compatibility
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    # Drop datetime columns (Streamlit can struggle displaying them sometimes)
    df = df.drop(columns=df.select_dtypes(include=['datetime64']).columns)

    # Drop constant columns
    df = df.loc[:, df.nunique() > 1]

    # Drop high-cardinality columns except target (likely IDs)
    high_card_cols = [col for col in df.columns if df[col].nunique() > 30 and col != target_column]
    df = df.drop(columns=high_card_cols)

    # One-hot encode categorical except target
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    cat_cols = [c for c in cat_cols if c != target_column]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Fill NaNs with median except target
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


# -------- Streamlit UI & Logic --------

st.title("🔮 Hybrid DNN-EQIC Clustering & Prediction")

uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    # Load dataset
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = df.columns.astype(str)  # ensure string col names
    st.subheader("Dataset Preview")
    st.dataframe(df.astype(str).head())

    # --- Ask user about train/test split ---
    auto_split = st.radio("Auto train/test split?", options=["Yes", "No"], index=0)

    if auto_split == "No":
        split_col = st.selectbox("Select column for train/test split range", options=df.columns)
        min_val = int(df[split_col].min())
        max_val = int(df[split_col].max())
        train_start = st.number_input(f"Train start value in '{split_col}'", min_value=min_val, max_value=max_val, value=min_val)
        train_end = st.number_input(f"Train end value in '{split_col}'", min_value=min_val, max_value=max_val, value=min_val + (max_val - min_val)//3)
        # Ensure valid range
        if train_start > train_end:
            st.error("Train start must be less than or equal to train end!")
            st.stop()
        train_mask = (df[split_col] >= train_start) & (df[split_col] <= train_end)
        train_df = df.loc[train_mask]
        test_df = df.loc[~train_mask]
        st.write(f"Training dataset size: {train_df.shape[0]} rows")
        st.write(f"Testing dataset size: {test_df.shape[0]} rows")
    else:
        # Auto split 70/30 by row index
        train_size = int(len(df)*0.7)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        st.write(f"Auto train/test split at row index {train_size}")
        st.write(f"Training dataset size: {train_df.shape[0]} rows")
        st.write(f"Testing dataset size: {test_df.shape[0]} rows")

    # --- Target variable and params ---
    target_column = st.selectbox("Select Target Column to Predict", options=df.columns)

    threshold = 0.5
    if target_column:
        # Target type detection
        target_unique = df[target_column].dropna().unique()
        if df[target_column].dtype == 'O' or len(target_unique) <= 10:
            target_type = 'binary' if len(target_unique) == 2 else 'categorical'
        else:
            target_type = 'continuous'
        st.write(f"Detected target type: `{target_type}`")

        if target_type == 'binary':
            threshold = st.slider("Select Threshold for Binary Prediction", 0.0, 1.0, 0.5)

        n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=6, value=3)

        # Preprocess train data
        train_clean = preprocess_df(train_df, target_column)
        features = [col for col in train_clean.columns if col != target_column]

        # Scale train data
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_clean[features])

        # Hyperparameter grid
        param_grid = {
            'alpha': [1.0, 2.0],
            'beta': [0.3, 0.5],
            'gamma': [0.7, 0.9],
            'kappa': [0.05, 0.1]
        }

        st.info("⏳ Searching for best hyperparameters on training data...")
        labels_train, best_params, best_score = hyperparameter_search(train_scaled, n_clusters=n_clusters, param_grid=param_grid)
        st.success(f"✅ Best Silhouette Score: {best_score:.4f}")
        st.json(best_params)

        # Final clustering on train data with best params
        labels_train, centroids, weights = enhanced_quantum_clustering(train_scaled, n_clusters=n_clusters, **best_params)
        train_clean["Cluster"] = labels_train

        # Map cluster to predicted target values on train
        if target_type == 'binary':
            cluster_map = train_clean.groupby('Cluster')[target_column].mean().to_dict()
            train_clean['Predicted'] = [int(cluster_map[l] > threshold) for l in labels_train]
        elif target_type == 'continuous':
            cluster_map = train_clean.groupby('Cluster')[target_column].mean().to_dict()
            train_clean['Predicted'] = [cluster_map[l] for l in labels_train]
        else:
            cluster_map = train_clean.groupby('Cluster')[target_column].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
            train_clean['Predicted'] = [cluster_map[l] for l in labels_train]

        st.subheader("📈 Training Data Cluster Analysis")
        st.write("Cluster Counts:")
        st.dataframe(train_clean['Cluster'].value_counts().rename("Count"))

        st.write("Cluster Means & Std Dev:")
        st.dataframe(train_clean.groupby('Cluster')[features + ['Predicted']].agg(['mean','std']))

        # Calculate centroid distances (train)
        centroid_distances = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                centroid_distances[i, j] = np.linalg.norm(centroids[i] - centroids[j])
        st.write("Centroid Euclidean Distances Matrix:")
        st.dataframe(pd.DataFrame(centroid_distances, index=[f"C{i}" for i in range(n_clusters)], columns=[f"C{j}" for j in range(n_clusters)]))

        # PCA plot train
        try:
            pca = PCA(n_components=2)
            train_pca = pca.fit_transform(train_scaled)
            fig, ax = plt.subplots(figsize=(8,6))
            for cluster in np.unique(labels_train):
                idx = labels_train == cluster
                ax.scatter(train_pca[idx, 0], train_pca[idx, 1], label=f"Cluster {cluster}", alpha=0.6)
            ax.set_title("PCA Projection of Training Clusters")
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"PCA plot not available: {e}")

        # --- Process test data similarly ---
        st.subheader("➡️ Applying Model to Test Data")
        test_clean = preprocess_df(test_df, target_column)
        test_clean = test_clean.reindex(columns=train_clean.columns.drop(['Cluster','Predicted']), fill_value=0)  # Ensure same columns

        # Scale test data using train scaler
        test_scaled = scaler.transform(test_clean[features])

        # Assign clusters to test
        labels_test = assign_clusters(test_scaled, centroids, weights, **best_params)
        test_clean["Cluster"] = labels_test

        # Predict target for test using cluster_map from train
        if target_type == 'binary':
            test_clean['Predicted'] = [int(cluster_map.get(l,0) > threshold) for l in labels_test]
        elif target_type == 'continuous':
            test_clean['Predicted'] = [cluster_map.get(l,np.nan) for l in labels_test]
        else:
            test_clean['Predicted'] = [cluster_map.get(l,np.nan) for l in labels_test]

        st.write(f"Test dataset size: {test_clean.shape[0]} rows")

        st.write("Test Cluster Counts:")
        st.dataframe(test_clean['Cluster'].value_counts().rename("Count"))

        st.write("Test Cluster Means & Std Dev:")
        st.dataframe(test_clean.groupby('Cluster')[features + ['Predicted']].agg(['mean','std']))

        # PCA plot test
        try:
            test_pca = pca.transform(test_scaled)
            fig2, ax2 = plt.subplots(figsize=(8,6))
            for cluster in np.unique(labels_test):
                idx = labels_test == cluster
                ax2.scatter(test_pca[idx, 0], test_pca[idx, 1], label=f"Cluster {cluster}", alpha=0.6)
            ax2.set_title("PCA Projection of Test Clusters")
            ax2.set_xlabel("PC 1")
            ax2.set_ylabel("PC 2")
            ax2.legend()
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"PCA test plot not available: {e}")

        # Show sample outputs
        st.subheader("🔍 Sample Outputs")
        st.write("Training Data (showing target, cluster, predicted, and features):")
        st.dataframe(train_clean[[target_column, 'Cluster', 'Predicted'] + features].astype(str).head(10))

        st.write("Test Data (showing target, cluster, predicted, and features):")
        st.dataframe(test_clean[[target_column, 'Cluster', 'Predicted'] + features].astype(str).head(10))

        # Download button for combined output
        combined_out = pd.concat([train_clean.assign(DataSet='Train'), test_clean.assign(DataSet='Test')])
        csv = combined_out.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Combined Results as CSV", data=csv, file_name="hybrid_dnn_eqic_results.csv", mime="text/csv")

