import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from datetime import datetime

# ---------------- Memory Structures -------------------
class EpisodicMemory:
    def __init__(self):
        self.episodes = {}

    def store(self, timestamp, pattern):
        self.episodes[timestamp] = pattern

class SemanticMemory:
    def __init__(self):
        self.associations = {}

    def update(self, feature_pair, strength):
        key = tuple(sorted(feature_pair))
        self.associations[key] = self.associations.get(key, 0.0) + strength

class HybridImputerPredictor:
    def __init__(self):
        self.memory = EpisodicMemory()
        self.semantics = SemanticMemory()
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = MinMaxScaler()
        self.trained = False

    def fit(self, df):
        self.feature_names = df.columns
        df_encoded = self._encode_categoricals(df.copy())
        df_imputed = pd.DataFrame(self.imputer.fit_transform(df_encoded), columns=df.columns)
        df_scaled = pd.DataFrame(self.scaler.fit_transform(df_imputed), columns=df.columns)

        for i, row in df_scaled.iterrows():
            self.memory.store(str(i), row.values)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    strength = np.abs(row[i] - row[j])
                    self.semantics.update((df.columns[i], df.columns[j]), 1 - strength)

        self.trained = True
        self.train_data = df
        self.train_scaled = df_scaled

    def predict_missing(self, df):
        df_encoded = self._encode_categoricals(df.copy())

        if df_encoded.empty:
            return df_encoded  # Skip if empty

        df_imputed = pd.DataFrame(self.imputer.transform(df_encoded), columns=df.columns)
        df_scaled = pd.DataFrame(self.scaler.transform(df_imputed), columns=df.columns)

        predictions = []
        for i, row in df_scaled.iterrows():
            if row.isnull().any():
                pred = row.copy()
                for idx, val in enumerate(row):
                    if np.isnan(val):
                        pred[idx] = self._predict_column_value(idx, row)
                predictions.append(pred)
            else:
                predictions.append(row)
        return pd.DataFrame(self.scaler.inverse_transform(predictions), columns=df.columns)

    def _predict_column_value(self, idx, row):
        known_mask = ~np.isnan(row)
        similarities = []
        for mem in self.memory.episodes.values():
            sim = -np.linalg.norm(row[known_mask] - mem[known_mask])
            similarities.append(sim)
        best_idx = np.argmax(similarities)
        best_match = list(self.memory.episodes.values())[best_idx]
        return best_match[idx]

    def _encode_categoricals(self, df):
        for col in df.select_dtypes(include='object'):
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        return df

# ---------------- Streamlit Dashboard -------------------
st.set_page_config(layout="wide")
st.title("🧠 Hybrid Brain-like Predictor and Analyzer")

st.sidebar.markdown("### Upload Your Excel File")
file = st.sidebar.file_uploader("Choose a file", type=['xlsx'])

if file:
    sheet_names = pd.ExcelFile(file).sheet_names
    sheet = st.sidebar.selectbox("Choose sheet", sheet_names)
    raw_df = pd.read_excel(file, sheet_name=sheet)

    st.markdown("### 📊 Raw Data Preview")
    st.dataframe(raw_df.head())

    st.markdown("---")
    st.markdown("### 🧹 Data Preprocessing")
    df_clean = raw_df.copy()
    df_clean = df_clean.dropna(axis=1, how='all')
    df_clean = df_clean.select_dtypes(include=[np.number, 'object'])

    predictor = HybridImputerPredictor()

    if st.button("Train Brain with Complete Data"):
        df_train = df_clean.dropna()
        predictor.fit(df_train)
        st.success("Training complete!")

        st.markdown("### 🧠 Memory Evolution")
        mem_df = pd.DataFrame(predictor.memory.episodes).T
        st.dataframe(mem_df.head())

        st.markdown("### 📈 Semantic Relationships")
        assoc_df = pd.DataFrame(predictor.semantics.associations.items(), columns=["Feature Pair", "Strength"])
        assoc_df = assoc_df.sort_values(by="Strength", ascending=False)
        st.dataframe(assoc_df.head(10))

    if predictor.trained:
        st.markdown("---")
        st.markdown("### 🔍 Predict Missing Values")
        df_test = df_clean[df_clean.isna().any(axis=1)]

        if df_test.empty:
            st.warning("🚫 No rows found with missing values in the target column.")
            st.stop()

        predicted = predictor.predict_missing(df_test)
        st.dataframe(predicted.head())

        st.markdown("### 📊 Dataset Statistics")
        st.markdown("#### Complete Data")
        st.dataframe(df_train.describe())

        st.markdown("#### Test Data with Predictions")
        st.dataframe(predicted.describe())

        st.markdown("---")
        st.markdown("### 🧬 Correlation Heatmap")
        corr = df_train.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("### 🔗 Cluster Analysis")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(predictor.train_scaled)
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X_pca)
        silhouette = silhouette_score(X_pca, kmeans.labels_)

        fig2, ax2 = plt.subplots()
        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis')
        ax2.set_title(f"KMeans Clusters (Silhouette Score = {silhouette:.2f})")
        st.pyplot(fig2)

else:
    st.info("Please upload an Excel file to begin analysis.")
