import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow model prediction
def model_prediction(test_img):
    model = tf.keras.model.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_img, target_size = (128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single img to batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

#Home
if(app_mode == 'Home'):
    st.header("Plant Disease Recognition System")