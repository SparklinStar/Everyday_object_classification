import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import style


page_bg_img = style.stylespy()  # used for styling the page

# Appname
st.set_page_config(page_title="Everyday Object Classifier", layout="wide")

st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #fff;'>Everyday Object Classifier</h1>", unsafe_allow_html=True)


# Load your model and its weights
model = tf.keras.models.load_model('dailyobjectsmodel.h5')
class_names = ["Big Truck","City Car","Multi Purpose Vehicle","Sedan","Sport Utility Vehicle","Truck","Van","cat","caterpillar","cockroach","cow","dog","rat","refrigerators","sofas","tvs","wall-clocks"
]  # List of your class names

# Define the Streamlit app
def main():
    st.write("Upload an image for classification")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        image = image.resize((224, 224))
        image = np.array(image)
        image = preprocess_input(image)

        # Make predictions
        predictions = model.predict(np.expand_dims(image, axis=0))
    
        if np.isnan(predictions).any():
         st.write("Prediction result is NaN. Please try with another image")
        else:
         predicted_class = np.argmax(predictions)
         confidence = predictions[0][predicted_class]

         st.write(f"Predicted class: {class_names[predicted_class]}")
         st.write(f"Confidence: {confidence:.2f}")

    items = [
    'dog', 'rat', 'caterpillar', 'cow', 'cockroach', 'cat', 'Van',
    'Truck', 'Big Truck', 'Multi Purpose Vehicle', 'Sedan',
    'Sport Utility Vehicle', 'City Car', 'tvs', 'refrigerators',
    'sofas', 'wall-clocks'
    ]

    st.title("This model is capable of classifying:")
    for item in items:
        st.write("- " + item)
# Run the app
if __name__ == '__main__':
    main()
