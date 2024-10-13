import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌺", layout="wide")

# Custom CSS
st.markdown("""
<style>
    body {
        color: #FFFFFF;
        background-color: #000000;
    }
    .stApp {
        background-color: #000000;
    }
    .stHeader {
        background-color: #4DB6AC;
        padding: 1rem;
        border-radius: 10px;
    }
    .stSubheader, h1, h2, h3, h4, h5, h6 {
        color: #4DB6AC;
    }
    .stSlider > div > div > div {
        background-color: #4DB6AC;
    }
    .stDataFrame {
        background-color: #333333;
    }
    .stPlotlyChart {
        background-color: #333333;
    }
    .stMarkdown {
        color: #FFFFFF;
    }
    .stSelectbox label, .stSlider label {
        color: #4DB6AC;
    }
    .stButton > button {
        color: #FFFFFF;
        background-color: #4DB6AC;
    }
    .stButton > button:hover {
        color: #FFFFFF;
        background-color: #80CBC4;
    }
    .stSidebar .sidebar-content {
        color: #FFFFFF;
    }
    .stSlider > div > div > div > div, .stSlider > div > div > div > div > div {
        color: blue;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; color: #4DB6AC;'>🌸 Iris Flower Prediction App 🌸</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("""
This app predicts the Iris flower species based on sepal and petal measurements.
Use the sliders in the sidebar to input flower measurements and see the prediction!
""")

# Sidebar
st.sidebar.header("🌿 Input Parameters")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width,
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 User Input Parameters")
    st.dataframe(df.style.highlight_max(axis=0))

# Load and train model
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

clf = RandomForestClassifier()
clf.fit(X, y)

# Make prediction
pred = clf.predict(df)
pred_prob = clf.predict_proba(df)

with col2:
    st.subheader("🔮 Prediction")
    st.write(f"The predicted Iris species is: **{iris.target_names[pred[0]]}**")
    
    st.subheader("🎯 Prediction Probability")
    prob_df = pd.DataFrame(pred_prob, columns=iris.target_names)
    st.dataframe(prob_df.style.format("{:.2%}").background_gradient(cmap="viridis"))

# Visualization
st.subheader("📈 Feature Importance")
feature_importance = pd.DataFrame({"feature": iris.feature_names, "importance": clf.feature_importances_})
fig = px.bar(feature_importance, x="importance", y="feature", orientation="h",
             title="Feature Importance in Prediction",
             labels={"importance": "Importance", "feature": "Feature"},
             color="importance", color_continuous_scale="viridis")
st.plotly_chart(fig)

# Dataset Overview
st.subheader("📚 Iris Dataset Overview")
tab1, tab2 = st.tabs(["📊 Data Sample", "📈 Pairplot"])

with tab1:
    st.dataframe(X.head())

with tab2:
    @st.cache_data
    def load_pairplot():
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        fig = px.scatter_matrix(df, dimensions=iris.feature_names, color="species")
        return fig

    st.plotly_chart(load_pairplot())

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit")
