from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import time 

def load_data(): 
    with st.spinner("Loading and Processing Data..."): time.sleep(2)

    anime_path = "data\data-anime.csv"
    raw_anime_df = pd.read_csv(anime_path)
    anime_df = raw_anime_df.copy()

    numeric_cols = ["Episodes", "Score", "Vote", "Ranked", "Popularity", "Duration"]
    for col in numeric_cols: 
        anime_df[col] = pd.to_numeric(anime_df[col], errors='coerce')
        anime_df[col].fillna(anime_df[col].median(), inplace=True)

    anime_df = anime_df.dropna(subset=["Score", "Vote"])

    anime_df = anime_df[["Vote", "Score", "Episodes", "Ranked", "Popularity", "Duration"]]

    categorical_features = ["Status", "Studios", "Source", "Rating"]
    existing_categorical_features = [col for col in categorical_features if col in anime_df.columns]
    if existing_categorical_features:
        anime_df = pd.get_dummies(anime_df, columns=existing_categorical_features, drop_first=True)

    anime_df.to_csv("anime_processed.csv", index=False)

    return raw_anime_df, anime_df

def train_fnn():
    df = pd.read_csv("anime_processed.csv")

    X = df.drop(columns=["Score"])
    y = df["Score"]

    imputer = SimpleImputer(strategy="mean") 
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)

    # Train the model
    with st.spinner("Training the FNN model..."):
        model.fit(X_train, y_train)

    st.success("Model Training Complete!")

    st.subheader("Training Progress")
    st.write(f"Training Score: {model.score(X_train, y_train):.4f}")
    st.write(f"Test Score: {model.score(X_test, y_test):.4f}")

    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_pred - y_test))
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")

def train_cnn():
    df = pd.read_csv("anime_processed.csv")

    X = df.drop(columns=["Score"])
    y = df["Score"]

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=200, random_state=42)

    # Train the model
    with st.spinner("Training the CNN model..."):
        model.fit(X_train, y_train)

    st.success("Model Training Complete!")

    st.subheader("Training Progress")
    st.write(f"Training Score: {model.score(X_train, y_train):.4f}")
    st.write(f"Test Score: {model.score(X_test, y_test):.4f}")

    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_pred - y_test))
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")

# Streamlit
def display():
    st.write("# Neural Network Models")
    raw_anime_df, anime_df = load_data()
    
    st.write("### Data Preview")
    st.write(raw_anime_df.head())

    st.write("### Processed Data Preview")
    st.write(anime_df.head())

    model_type = st.selectbox("Select Model Type", ("FNN", "CNN"))

    if st.button("Show Model Results"):
        st.write(f"## {model_type} Model Performance")
        
        if model_type == "FNN":
            train_fnn()
        elif model_type == "CNN":
            train_cnn()

if __name__ == "__main__":
    display()