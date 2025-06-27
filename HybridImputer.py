import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.cluster import KMeans
from datetime import datetime

# --- Quantum-Inspired Hybrid Unit (Neuron) ---
class HybridUnit:
    def __init__(self, pattern, learning_rate=0.1):
        self.pattern = np.array(pattern)
        self.learning_rate = learning_rate
        self.usage_count = 0
        self.created_at = datetime.now()

    def quantum_similarity(self, input_pattern):
        diff = np.abs(input_pattern - self.pattern)
        euclidean = np.sqrt(np.sum(diff ** 2))
        return np.exp(-2.0 * euclidean) + 0.5 / (1 + 0.9 * euclidean)

    def adapt(self, input_pattern):
        self.pattern += self.learning_rate * (input_pattern - self.pattern)
        self.usage_count += 1

# --- Memory Structures ---
class SemanticMemory:
    def __init__(self):
        self.relationships = {}

    def update(self, pattern):
        for i in range(len(pattern)):
            for j in range(i+1, len(pattern)):
                key = tuple(sorted((i, j)))
                if key not in self.relationships:
                    self.relationships[key] = 0
                self.relationships[key] += pattern[i] * pattern[j]

class EpisodicMemory:
    def __init__(self):
        self.episodes = []

    def store(self, input_vector):
        self.episodes.append((datetime.now(), input_vector))

# --- Core Predictor ---
class HybridImputerPredictor:
    def __init__(self, quantum_threshold=0.6):
        self.units = []
        self.quantum_threshold = quantum_threshold
        self.semantic_memory = SemanticMemory()
        self.episodic_memory = EpisodicMemory()

    def _match_unit(self, input_vector):
        best_similarity = 0
        best_unit = None
        for unit in self.units:
            sim = unit.quantum_similarity(input_vector)
            if sim > best_similarity:
                best_similarity = sim
                best_unit = unit
        return best_unit, best_similarity

    def learn(self, input_vector):
        self.episodic_memory.store(input_vector)
        unit, similarity = self._match_unit(input_vector)

        if unit and similarity >= self.quantum_threshold:
            unit.adapt(input_vector)
        else:
            self.units.append(HybridUnit(input_vector))

        self.semantic_memory.update(input_vector)

    def predict_missing(self, input_vector):
        nan_indices = np.where(np.isnan(input_vector))[0]
        if len(nan_indices) == 0:
            return input_vector

        unit, similarity = self._match_unit(input_vector)
        if not unit:
            return input_vector

        imputed = input_vector.copy()
        for idx in nan_indices:
            imputed[idx] = unit.pattern[idx]
        return imputed

    def explain_relationships(self):
        sorted_rel = sorted(self.semantic_memory.relationships.items(), key=lambda x: -x[1])
        return [(f"Feature {i} & {j}", score) for ((i, j), score) in sorted_rel[:10]]

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("🧠 Quantum-Inspired Predictor & Analyzer")

st.sidebar.header("Upload Dataset")
file = st.sidebar.file_uploader("Choose Excel file", type=['xlsx'])

if file:
    sheet_names = pd.ExcelFile(file).sheet_names
    sheet = st.sidebar.selectbox("Select Sheet", sheet_names)
    df = pd.read_excel(file, sheet_name=sheet)
    df = df.select_dtypes(include=[np.number])

    st.subheader("Raw Data")
    st.dataframe(df.head())

    target_variable = st.sidebar.selectbox("Select Variable to Predict (must contain NaNs)", df.columns)
    train_df = df.dropna()
    test_df = df[df[target_variable].isna()]

    imputer = SimpleImputer(strategy='mean')
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(imputer.fit_transform(train_df))

    predictor = HybridImputerPredictor()
    for row in train_scaled:
        predictor.learn(row)

    st.subheader("Memory Evolution")
    st.write(f"Total memory episodes: {len(predictor.episodic_memory.episodes)}")

    st.subheader("Semantic Relationships")
    rel_df = pd.DataFrame(predictor.explain_relationships(), columns=["Feature Pair", "Strength"])
    st.dataframe(rel_df)

    if not test_df.empty:
        test_imputed = imputer.transform(test_df)
        test_scaled = scaler.transform(test_imputed)
        predictions = []
        for row in test_scaled:
            pred = predictor.predict_missing(row)
            predictions.append(pred)
        pred_df = pd.DataFrame(scaler.inverse_transform(predictions), columns=df.columns)
        st.subheader("Predicted Missing Values")
        st.dataframe(pred_df[[target_variable]].head())

    st.subheader("Descriptive Statistics")
    st.write("**Training Set**")
    st.dataframe(train_df.describe())

    if not test_df.empty:
        st.write("**Predicted Test Set**")
        st.dataframe(pred_df.describe())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(train_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Cluster Visualization (KMeans + PCA)")
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(train_scaled)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(reduced)
    silhouette = silhouette_score(reduced, kmeans.labels_)
    fig2, ax2 = plt.subplots()
    ax2.scatter(reduced[:, 0], reduced[:, 1], c=kmeans.labels_, cmap='viridis')
    ax2.set_title(f"Clusters (Silhouette Score: {silhouette:.2f})")
    st.pyplot(fig2)

else:
    st.info("Please upload an Excel file to start analysis.")
