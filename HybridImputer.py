import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

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

# --- Semantic Memory for Feature Associations ---
class SemanticMemory:
    def __init__(self):
        self.relationships = {}  # (i,j) -> avg co-activation

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

# --- HybridImputerPredictor Core ---
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

    def explain_relationships(self, feature_names):
        sorted_rel = sorted(self.semantic_memory.relationships.items(), key=lambda x: -x[1])
        return [(f"{feature_names[i]} & {feature_names[j]}", score) for ((i, j), score) in sorted_rel[:10]]

# ---------------- Streamlit Dashboard -------------------
st.set_page_config(layout="wide")
st.title("🧠 Quantum-Inspired Brain-like Predictor")

st.sidebar.header("Upload Excel Dataset")
file = st.sidebar.file_uploader("Choose a file", type=['xlsx'])

if file:
    sheet_names = pd.ExcelFile(file).sheet_names
    sheet = st.sidebar.selectbox("Select sheet", sheet_names)
    df_raw = pd.read_excel(file, sheet_name=sheet)

    st.markdown("### Raw Data Preview")
    st.dataframe(df_raw.head())

    df = df_raw.select_dtypes(include=[np.number]).copy()
    feature_names = df.columns.tolist()
    imputer = SimpleImputer(strategy='mean')
    scaler = MinMaxScaler()

    df_complete = df.dropna()
    df_scaled = scaler.fit_transform(imputer.fit_transform(df_complete))

    predictor = HybridImputerPredictor()
    for row in df_scaled:
        predictor.learn(row)

    st.success("Model trained with complete rows!")

    st.markdown("### 🧬 Top Semantic Relationships")
    relationships = predictor.explain_relationships(feature_names)
    for rel, score in relationships:
        st.write(f"{rel}: {score:.4f}")

    st.markdown("### 🔮 Predict Missing Values")
    df_test = df[df.isna().any(axis=1)].copy()
    if df_test.empty:
        st.info("No missing values detected for prediction.")
    else:
        df_test_imputed = imputer.transform(df_test)
        df_test_scaled = scaler.transform(df_test_imputed)
        predictions = [predictor.predict_missing(row) for row in df_test_scaled]
        df_predicted = pd.DataFrame(scaler.inverse_transform(predictions), columns=feature_names)

        st.markdown("#### Predicted Values")
        st.dataframe(df_predicted.head())

    st.markdown("### 📊 Heatmap of Complete Data")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_complete.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

else:
    st.info("Upload an Excel file to begin.")
