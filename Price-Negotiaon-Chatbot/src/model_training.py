from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from nlp_utils import preprocess_text
import pandas as pd

# Initialize the vectorizer
vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
def train_negotiation_model(train_data):
    # Preprocess the data
    train_data = preprocess_train_data(train_data)
    
    # Initialize the vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the training data
    X = vectorizer.fit_transform(train_data['customer_input'])
    y = train_data['response']

    # Initialize and train the model
    model = LogisticRegression()
    model.fit(X, y)

    # Save the trained model and vectorizer
    with open("D:\\VerveBridge\\ChatBot\\models\\negotiation_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    with open("D:\\VerveBridge\\ChatBot\\models\\vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)

def preprocess_train_data(train_data):
    # Flatten lists to strings if necessary
    if isinstance(train_data['customer_input'].iloc[0], list):
        train_data['customer_input'] = [' '.join(item) for item in train_data['customer_input']]
    elif isinstance(train_data['customer_input'].iloc[0], str):
        pass
    else:
        raise ValueError("The 'customer_input' column should contain strings or lists of strings.")

    return train_data


if __name__ == "__main__":
    # Load preprocessed data
    with open('D:\\VerveBridge\\ChatBot\\data\\processed_data.pkl', 'rb') as f:
        train_data, _ = pickle.load(f)
    
    train_negotiation_model(train_data)
