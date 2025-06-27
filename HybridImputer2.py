import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- Quantum-Inspired Hybrid Unit ---
class HybridUnit:
    def __init__(self, pattern, learning_rate=0.1):
        self.pattern = np.array(pattern)
        self.learning_rate = learning_rate
        self.usage_count = 0
        self.created_at = datetime.now()

    def quantum_similarity(self, input_pattern):
        diff = np.abs(input_pattern - self.pattern)
        dist = np.sqrt(np.sum(diff ** 2))
        return np.exp(-2.0 * dist) + 0.5 / (1 + 0.9 * dist)

    def adapt(self, input_pattern):
        self.pattern += self.learning_rate * (input_pattern - self.pattern)
        self.usage_count += 1

# --- Semantic Memory ---
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

# --- Episodic Memory ---
class EpisodicMemory:
    def __init__(self):
        self.episodes = []

    def store(self, input_vector):
        self.episodes.append((datetime.now(), input_vector))

# --- Hybrid Predictor ---
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
        return sorted(self.semantic_memory.relationships.items(), key=lambda x: -x[1])

# --- Streamlit Interface ---
st.set_page_config(layout="wide")
st.title("🧠 Quantum-Inspired Brain-like Predictor")

file = st.sidebar.file_uploader("Upload Excel File", type=['xlsx'])
if file:
    sheet_names = pd.ExcelFile(file).sheet_names
    sheet = st.sidebar.selectbox("Select Sheet", sheet_names)
    df = pd.read_excel(file, sheet_name=sheet)
    st.markdown("### Raw Data")
    st.dataframe(df.head())

    df_clean = df.copy().dropna(axis=1, how='all')
    df_clean = df_clean.select_dtypes(include=[np.number])
    target = st.selectbox("Select Target Column to Predict (Missing Values Required)", df_clean.columns)

    df_train = df_clean[df_clean[target].notna()]
    df_test = df_clean[df_clean[target].isna()]

    imputer = SimpleImputer(strategy='mean')
    scaler = MinMaxScaler()

    data_train = scaler.fit_transform(imputer.fit_transform(df_train))
    predictor = HybridImputerPredictor()

    for row in data_train:
        predictor.learn(row)

    st.success("Model Trained with Complete Rows")

    # --- Forecasting / Prediction ---
    st.markdown("### 🔍 Predicted Missing Values")
    predicted = []
    data_test = scaler.transform(imputer.transform(df_test))
    for row in data_test:
        predicted.append(predictor.predict_missing(row))

    predicted_df = pd.DataFrame(scaler.inverse_transform(predicted), columns=df_test.columns)
    st.dataframe(predicted_df.head())

    # --- Save Forecasted Output ---
    output_df = df.copy()
    output_df.loc[df_test.index, target] = predicted_df[target].values
    output_df.to_excel("predicted_output.xlsx", index=False)
    st.download_button("Download Forecasted Data", "predicted_output.xlsx")

    # --- Descriptive Statistics ---
    st.markdown("### 📊 Descriptive Statistics")
    st.dataframe(df_train.describe())
    st.dataframe(predicted_df.describe())

    # --- Correlation ---
    st.markdown("### 🔗 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_train.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # --- Clustering Visualization ---
    st.markdown("### 🧬 Cluster Visualization")
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data_train)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(reduced)
    score = silhouette_score(reduced, kmeans.labels_)
    fig2, ax2 = plt.subplots()
    ax2.scatter(reduced[:, 0], reduced[:, 1], c=kmeans.labels_, cmap='viridis')
    ax2.set_title(f"KMeans Clusters - Silhouette Score: {score:.2f}")
    st.pyplot(fig2)

    # --- Memory & Semantic Visualization ---
    st.markdown("### 🧠 Memory Evolution")
    st.write(f"Number of Stored Episodes: {len(predictor.episodic_memory.episodes)}")

    st.markdown("### 🧩 Top Feature Relationships")
    top_rels = predictor.explain_relationships()
    for ((i, j), strength) in top_rels[:10]:
        st.markdown(f"- Feature {df_clean.columns[i]} & {df_clean.columns[j]}: {strength:.4f}")

else:
    st.info("Please upload an Excel file to begin.")
