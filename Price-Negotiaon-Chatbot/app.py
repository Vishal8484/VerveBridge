import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# Load the model and vectorizer
with open("models/negotiation_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("models/vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Path to the chatbot image
image_path = "D:\VerveBridge\ChatBot\GIU AMA 255-07.jpg"  # Adjust this path as needed

# Streamlit app layout
st.title('Chatbot Price Negotiation')

# Display the chatbot image
st.image(image_path, caption='Chatbot', use_column_width=True)

# Text input from the user
user_input = st.text_input("Enter your query:")

# Handle user input
if st.button('Submit'):
    if user_input:
        # Preprocess and transform the input
        input_vector = vectorizer.transform([user_input])
        
        # Get the model's prediction
        prediction = model.predict(input_vector)
        
        # Display the response
        st.write(f"Response: {prediction[0]}")
    else:
        st.write("Please enter a query.")
