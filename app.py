import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Crop Disease Detector", page_icon="🍃")
st.title("🍃 Indian Crop Disease Detector")
st.markdown("### Final Year Project: B.Tech CSE")

# 2. Load the Model (Brain)
@st.cache_resource  # This keeps the model in memory so it doesn't reload every time
def load_my_model():
    return tf.keras.models.load_model('crop_disease_model.h5')

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. Disease Categories
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# 4. User Interface
file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocessing
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    # Prediction
    with st.spinner("Analyzing Leaf..."):
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        result = CLASS_NAMES[np.argmax(score)]
        confidence = 100 * np.max(score)
    
    # Results
    st.success(f"Prediction: **{result.replace('_', ' ')}**")
    st.progress(int(confidence))
    st.info(f"Confidence: **{confidence:.2f}%**")