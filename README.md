# Handwritten Digit Recognition System
This repository contains the code for a handwritten digit recognition system, implemented using TensorFlow and Keras. 

The system includes a Jupyter Notebook ('handwritten digit recognition system .ipynb') for training the neural network model on the MNIST dataset, covering data loading, preprocessing, model definition, training, and evaluation.  It also includes a Python script ('app.py') that uses Streamlit to create a web application.  This application allows users to upload an image of a handwritten digit, and the application will predict the digit.  

The notebook loads training data, preprocesses it by separating features and labels, normalizing pixel values, reshaping the data, and one-hot encoding the labels.  It then splits the data, defines a Keras model, compiles it with the Adam optimizer, trains it, evaluates it, saves it as 'my_model.keras', and predicts labels for test images.

The Streamlit application loads this trained model and defines a function to preprocess uploaded images by converting them to grayscale, resizing them, converting them to a NumPy array, reshaping, and normalizing them.  The main function sets up the application, allows image uploads, displays uploaded images, preprocesses them, predicts the digit using the model, and displays the prediction, handling any errors during processing.  

To use the system, clone the repository, install the required packages using pip (e.g., 'pip install tensorflow pandas matplotlib scikit-learn streamlit Pillow'), and then either run the notebook to train the model (optionally, ensuring the Train.csv file is available) or run the 'app.py' script using 'streamlit run app.py' to launch the prediction application.

## DATASET OBTAINED FROM:
https://www.kaggle.com/datasets/bhavikjikadara/handwritten-digit-recognition

## ALGORITHM USED:

feedforward neural network ; https://www.geeksforgeeks.org/feedforward-neural-network/

## LIBRARIES REQUIRED:
The system requires Python 3.x, TensorFlow, Keras, Pandas, Matplotlib, scikit-learn, Streamlit, and PIL (Pillow). 
