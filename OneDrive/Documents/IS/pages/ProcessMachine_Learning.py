import streamlit as st

def main():
    st.write("### Machine Learning")

    st.write("Import libraries for use in Machine learning")
    st.image("code_image\library_ML.png", caption="Import library", use_container_width=True)

    st.write("I used the dataset from https://www.kaggle.com/datasets/duongtruongbinh/manga-and-anime-dataset using the manga.csv file.")
    st.image("code_image\load_process_data_ML.png", caption="Load and Process data", use_container_width=True)

    st.image("code_image\Train_SVM_KNN_ML.png", caption="Train SVM and KNN models", use_container_width=True)

    st.image("code_image\preview_data_ML.png", caption="Visualize data", use_container_width=True)

    st.image("code_image\confusion_matrix_ML.png", caption="Plot the confusion matrix", use_container_width=True)

    st.image("code_image\display_ML.png", caption="Display on Streamlit", use_container_width=True)

    st.write("### SVM and KNN theories")

    st.write("### SVM")
    st.write("Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. The primary goal of SVM is to find the optimal hyperplane that best separates data points of different classes in a high-dimensional space. The hyperplane is chosen so that the margin between the closest points of different classes (called support vectors) is maximized. SVM can handle both linear and non-linear classification problems using kernel functions like linear, polynomial, and radial basis function (RBF) to transform the data into a higher-dimensional space where it becomes easier to separate.")
    st.write("https://scikit-learn.org/stable/modules/svm.html")

    st.write("### KNN")
    st.write("K-Nearest Neighbors (KNN) is a simple, instance-based supervised learning algorithm used for classification and regression. KNN works by identifying the 'k' nearest neighbors of a data point based on a distance metric (usually Euclidean distance) and classifying the point according to the majority class among the k neighbors. The value of 'k' determines how many neighbors influence the decision. KNN does not have a model-building phase; instead, it stores the entire training dataset and makes predictions based on the proximity of new data points to the training examples.")
    st.write("https://scikit-learn.org/stable/modules/neighbors.html")

    if st.button("Machine Learning Models"):
        st.switch_page("pages\Models_ML.py")

if __name__ == "__main__":
    main()

