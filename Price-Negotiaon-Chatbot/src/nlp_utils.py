import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

nltk.download('punkt')

def tokenize_text(text):
    return word_tokenize(text.lower())

def preprocess_text(text):
    """Tokenizes and preprocesses the input text."""
    return word_tokenize(text.lower())


def vectorize_text(texts):
    vectorizer = TfidfVectorizer(tokenizer=tokenize_text, token_pattern=None)  # Set token_pattern=None
    text_vectors = vectorizer.fit_transform(texts)
    
    return text_vectors, vectorizer  # Return the vectorizer along with text vectors

def save_vectorizer(vectorizer):
    # Ensure the models directory exists
    model_dir = "D:\\VerveBridge\\ChatBot\\models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save the vectorizer model
    with open(os.path.join(model_dir, 'nlp_model.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
