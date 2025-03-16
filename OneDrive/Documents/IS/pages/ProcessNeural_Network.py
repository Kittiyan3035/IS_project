import streamlit as st

def main():
    st.write("### Neural Network")

    st.write("Import libraries for use in Neural Network")
    st.image("code_image\library_NN.png", caption="Import library", use_container_width=True)

    st.write("I used the dataset from https://www.kaggle.com/datasets/duongtruongbinh/manga-and-anime-dataset using the file anime.csv but due to uploading issue with my file anime_path = 'data\ anime.csv' (\ a is an escape character) it failed to upload so I renamed the file to data-anime.csv.")
    st.image("code_image\load_process_data_NN.png", caption="Load and Process data", use_container_width=True)

    st.image("code_image\Train_FNN_NN.png", caption="Train the FNN model", use_container_width=True)

    st.image("code_image\Train_CNN_NN.png", caption="Train the CNN model", use_container_width=True)

    st.image("code_image\display_NN.png", caption="Display on Streamlit", use_container_width=True)

    st.write("### FVV and CNN theories")

    st.write("### Feedforward Neural Network (FNN)")
    st.write("A Feedforward Neural Network (FNN) is a type of artificial neural network where the connections between the nodes do not form a cycle. The network consists of three main layers: the input layer, one or more hidden layers, and the output layer. In an FNN, data flows in one direction—from the input layer to the hidden layers and then to the output layer—without any loops. Each node (neuron) in a layer is connected to every node in the previous and subsequent layers, and each connection has a weight. The network learns by adjusting these weights based on the error (difference between predicted and actual values) using a method like backpropagation and an optimization algorithm like gradient descent.")
    st.write("https://scikit-learn.org/stable/modules/neural_networks_supervised.html")

    st.write("### Convolutional Neural Network (CNN)")
    st.write("A Convolutional Neural Network (CNN) is a deep learning algorithm mainly used for image classification, recognition, and processing. CNNs are designed to automatically and adaptively learn spatial hierarchies of features through the use of convolutional layers, pooling layers, and fully connected layers. In the convolutional layer, a filter (or kernel) slides over the input image to extract features such as edges, textures, and patterns. Pooling layers reduce the spatial dimensions, helping in making the model invariant to small translations in the input. CNNs are particularly effective for tasks involving image data because they can reduce the number of parameters and capture local dependencies through convolutional operations.")
    st.write("https://www.tensorflow.org/tutorials/images/cnn")

    if st.button("Neural Network Models"):
        st.switch_page("pages\Models_NN.py")

if __name__ == "__main__":
    main()

