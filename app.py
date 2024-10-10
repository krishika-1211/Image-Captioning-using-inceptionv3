import streamlit as st
import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('model.h5')
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

with open('features.pkl', 'rb') as file:
    features = pickle.load(file)
    
# Function to convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to generate captions
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return final_caption

# Streamlit UI
st.title("Image Caption Generator")
st.write("Upload an image or provide the image path below:")

# Input for image path
image_path = st.text_input("Image Path:", "")

# Button to generate caption
if st.button("Generate Caption"):
    # Check if the image path is valid
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Extract features (adjust the feature extraction code as necessary)
        img_array = np.array(image.resize((299, 299)))
        img_array = np.expand_dims(img_array, axis=0)  # Reshape for the model
        features_extracted = model.predict(img_array)  # Use the appropriate feature extraction model

        # Generate the caption
        caption = predict_caption(model, features_extracted, tokenizer, max_length)
        st.write("Predicted Caption:", caption)
    else:
        st.error("Error: The provided image path does not exist.")
        
