import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
model = load_model('my_model.keras')  # Replace with your model path

def preprocess_image(image):
    img = image.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 28, 28, 1) / 255.0
    return img_array

def main():
    st.title('Handwritten Digit Recognition')

    uploaded_file = st.file_uploader("Upload an image of a digit", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_container_width=True)

            img_array = preprocess_image(image)
            prediction = model.predict(img_array)
            digit = np.argmax(prediction)

            st.write(f'## Predicted Digit: {digit}')

        except Exception as e:
            st.error(f'Error processing image: {e}')

if __name__ == '__main__':
    main()