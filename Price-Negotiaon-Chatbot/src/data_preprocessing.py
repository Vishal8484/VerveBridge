import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

def preprocess_data(pricing_data_path, negotiation_data_path):
    # Load and preprocess pricing data
    pricing_data = pd.read_csv("D:\VerveBridge\ChatBot\data\pricing_data.csv")
    negotiation_data = pd.read_csv("D:\VerveBridge\ChatBot\data\data_negotiation.csv")
    
    # Update to use ffill() directly instead of fillna(method='ffill')
    pricing_data.ffill(inplace=True)
    negotiation_data.ffill(inplace=True)
    
    # Split data for training and testing
    train_data, test_data = train_test_split(negotiation_data, test_size=0.2, random_state=42)
    
    # Save processed data
    with open('D:\\VerveBridge\\ChatBot\\data\\processed_data.pkl', 'wb') as f:
        pickle.dump((train_data, test_data), f)

    return train_data, test_data

if __name__ == "__main__":
    preprocess_data('../data/pricing_data.csv', '../data/negotiation_data.csv')
