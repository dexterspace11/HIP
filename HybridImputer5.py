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
    
    sum_weights = np.sum(weights)
    if sum_weights < epsilon:
        # Avoid division by zero by assigning uniform weights
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

    auto_split = st.radio("Automatically split train/test dataset?", options=["Yes", "No"], index=0)

    train_df = None
    test_df = None

    if auto_split == "No":
        split_column = st.selectbox("Select column for train/test split reference", options=df.columns)
        min_val = int(df[split_column].min())
        max_val = int(df[split_column].max())
        train_start = st.number_input(f"Train range start (inclusive) for column '{split_column}'", min_value=min_val, max_value=max_val, value=min_val)
        train_end = st.number_input(f"Train range end (inclusive) for column '{split_column}'", min_value=min_val, max_value=max_val, value=min_val + (max_val - min_val)//3)
        
        # Split based on column and range
        train_df = df[(df[split_column] >= train_start) & (df[split_column] <= train_end)]
        test_df = df[~((df[split_column] >= train_start) & (df[split_column] <= train_end))]
        st.write(f"Train dataset shape: {train_df.shape}")
        st.write(f"Test dataset shape: {test_df.shape}")
    else:
        train_ratio = st.slider("Train set ratio (for auto split)", min_value=0.1, max_value=0.9, value=0.7)
        train_size = int(len(df)*train_ratio)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        st.write(f"Train dataset shape: {train_df.shape}")
        st.write(f"Test dataset shape: {test_df.shape}")

    target_column = st.selectbox("Select Target Column to Predict", options=df.columns)

    threshold = st.slider("Threshold for Binary Prediction (only used if target is binary)", 0.0, 1.0, 0.5)

    n_clusters = st.slider("Select Number of Clusters", 2, 6, 3)

    start_analysis = st.button("Start Analysis")

    if start_analysis:
        # Preprocess train and test separately (using train target)
        train_clean = preprocess_df(train_df, target_column)
        test_clean = preprocess_df(test_df, target_column)

        # Align columns in test to train (handle one-hot columns)
        test_clean = test_clean.reindex(columns=train_clean.columns, fill_value=0)

        features = [col for col in train_clean.columns if col != target_column]

        # Scale data
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_clean[features])
        test_scaled = scaler.transform(test_clean[features])

        # Hyperparameter tuning on train
        param_grid = {
            'alpha': [1.0, 2.0],
            'beta': [0.3, 0.5],
            'gamma': [0.7, 0.9],
            'kappa': [0.05, 0.1]
        }

        st.info("⏳ Searching for best hyperparameters on train data...")
        labels_train, best_params, best_score = hyperparameter_search(train_scaled, n_clusters=n_clusters, param_grid=param_grid)
        st.success(f"✅ Best Silhouette Score (Train): {best_score:.4f}")
        st.json(best_params)

        # Final clustering on train
        labels_train, centroids, weights = enhanced_quantum_clustering(train_scaled, n_clusters=n_clusters, **best_params)
        train_clean["Cluster"] = labels_train

        # Predict clusters on test
        labels_test = assign_clusters(test_scaled, centroids, weights, **best_params)
        test_clean["Cluster"] = labels_test

        # Map clusters to target for prediction
        target_type = 'binary' if train_clean[target_column].dropna().nunique() == 2 else (
                      'categorical' if train_clean[target_column].dtype == 'O' or train_clean[target_column].nunique() <= 10 else 'continuous')

        def cluster_to_pred(df):
            if target_type == 'binary':
                cluster_map = df.groupby('Cluster')[target_column].mean().to_dict()
                return [int(cluster_map.get(l, 0) > threshold) for l in df['Cluster']]
            elif target_type == 'continuous':
                cluster_map = df.groupby('Cluster')[target_column].mean().to_dict()
                return [cluster_map.get(l, np.nan) for l in df['Cluster']]
            else:
                cluster_map = df.groupby('Cluster')[target_column].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).to_dict()
                return [cluster_map.get(l, np.nan) for l in df['Cluster']]

        train_clean['Predicted'] = cluster_to_pred(train_clean)
        test_clean['Predicted'] = cluster_to_pred(test_clean)

        # Show analysis for train and test separately
        st.subheader("📈 Train Dataset Cluster Analysis")
        st.write(train_clean.groupby('Cluster').agg({target_column: ['mean', 'std', 'count']}))
        st.write("Cluster counts:")
        st.write(train_clean['Cluster'].value_counts())

        st.subheader("📉 Test Dataset Cluster Analysis")
        st.write(test_clean.groupby('Cluster').agg({target_column: ['mean', 'std', 'count']}))
        st.write("Cluster counts:")
        st.write(test_clean['Cluster'].value_counts())

        # PCA visualization combined train+test
        st.subheader("🌀 PCA Projection of Combined Train and Test")
        combined_scaled = np.vstack([train_scaled, test_scaled])
        combined_labels = np.concatenate([labels_train, labels_test])
        pca = PCA(n_components=2)
        combined_pca = pca.fit_transform(combined_scaled)

        fig, ax = plt.subplots(figsize=(8,6))
        for cluster in np.unique(combined_labels):
            idx = combined_labels == cluster
            ax.scatter(combined_pca[idx, 0], combined_pca[idx, 1], label=f"Cluster {cluster}", alpha=0.6)
        ax.set_title("PCA Cluster Projection (Train + Test)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        st.pyplot(fig)

        # Centroid analysis - display centroid vectors (scaled)
        st.subheader("🔎 Centroid Feature Analysis")
        centroids_df = pd.DataFrame(centroids, columns=features)
        st.dataframe(centroids_df)

        # Prepare output for download - combine original data + clusters + predicted values
        output_train = train_df.copy()
        output_train['Cluster'] = train_clean['Cluster'].values
        output_train['Predicted_' + target_column] = train_clean['Predicted'].values

        output_test = test_df.copy()
        output_test['Cluster'] = test_clean['Cluster'].values
        output_test['Predicted_' + target_column] = test_clean['Predicted'].values

        combined_output = pd.concat([output_train, output_test])
        combined_output = combined_output.sort_index()

        csv = combined_output.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Dataset with Clusters and Predictions", data=csv, file_name="clustered_predictions.csv", mime="text/csv")
