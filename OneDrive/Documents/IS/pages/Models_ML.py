from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import time 

def load_data(): 
    
    with st.spinner("Loading and Processing Data..."):
        time.sleep(2)

    manga_path = "data/manga.csv"
    raw_manga_df = pd.read_csv(manga_path)
    manga_df = raw_manga_df.copy()

    manga_df.replace('Unknown', np.nan, inplace=True)
    for col in ["Genres", "Themes", "Demographics"]:
        manga_df[col].fillna("Unknown", inplace=True)

    numeric_cols = ["Members", "Favorite", "Volumes", "Chapters"]
    for col in numeric_cols: 
        manga_df[col] = pd.to_numeric(manga_df[col], errors='coerce')
        manga_df[col].fillna(manga_df[col].median(), inplace=True)

    manga_df["Genres"] = manga_df["Genres"].apply(lambda x: x.split(",")[0] if isinstance(x, str) else "Unknown")

    drop_cols = ["Title", "Published", "Serialization", "Author"]
    manga_df.drop(columns=drop_cols, inplace=True)
    manga_df.dropna(subset=["Popularity"], inplace=True)

    label_encoders = {}
    
    le_genres = LabelEncoder()
    manga_df["Genres"] = le_genres.fit_transform(manga_df["Genres"])
    label_encoders["Genres"] = le_genres

    for col in ["Status", "Themes", "Demographics"]:
        le = LabelEncoder()
        manga_df[col] = le.fit_transform(manga_df[col])
        label_encoders[col] = le 

    return raw_manga_df, manga_df, label_encoders

def train_models(df, model_type):
    X = df.drop(columns=["Popularity"])
    y = df["Genres"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM Model
    if model_type == "SVM":
        model = SVC(kernel="rbf", C=1.0)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    # Train KNN Model
    elif model_type == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    return X_test, y_test, y_pred, model

def visualize_data(df, label_encoders):
    st.write("## Genres Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))

    genre_names = label_encoders["Genres"].inverse_transform(df["Genres"])

    ax.hist(genre_names, bins=len(np.unique(genre_names)), color="skyblue", edgecolor="black", alpha=0.7)
    ax.set_title("Genres Distribution")
    ax.set_xlabel("Genres")
    ax.set_ylabel("Count")
    plt.xticks(rotation=90)
    st.pyplot(fig)

def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)
    
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    
    classes = np.unique(y_test)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white")

    st.pyplot(fig)
    
def display():
    st.write("# Machine Learning Models")
    raw_manga_df, manga_df, label_encoders = load_data()

    st.write("### Data Preview")
    st.write(raw_manga_df.head())

    st.write("### Processed Data Preview")
    st.write(manga_df.head())

    model_type = st.selectbox("Select Model Type", ("SVM", "KNN"))

    if st.button("Show Model Results"):
        X_test, y_test, y_pred, model = train_models(manga_df, model_type)

        visualize_data(manga_df, label_encoders)

        accuracy = accuracy_score(y_test, y_pred)
        precision = classification_report(y_test, y_pred, output_dict=True)["accuracy"]
        recall = classification_report(y_test, y_pred, output_dict=True)["macro avg"]["recall"]
        f1 = classification_report(y_test, y_pred, output_dict=True)["macro avg"]["f1-score"]
        
        st.write(f"## {model_type} Model Performance")
        st.write(f"**Accuracy**: {accuracy:.4f}")
        st.write(f"**Precision**: {precision:.4f}")
        st.write(f"**Recall**: {recall:.4f}")
        st.write(f"**F1-Score**: {f1:.4f}")

        plot_confusion_matrix(y_test, y_pred, model_type)


if __name__ == "__main__":
    display()