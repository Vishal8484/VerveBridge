import pandas as pd
import joblib 


#!pip install --upgrade astrapy

from astrapy import DataAPIClient

collection_name = "books"

cursor = db.get_collection("books").find()

import pandas as pd
df = pd.DataFrame(list(cursor))

print("Dataset Information:")
print(df.info())

print("\nFirst Few Rows of the Dataset:")
print(df.head())

print("Summary Statistics:")
print(df.describe())

print("\nNumber of Unique Values:")
print(df.nunique())

missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values)


def preprocess_data(df):
    df.fillna('', inplace=True)
    df['combined_features'] = df['Title'] + ' ' + df['Author'] + ' ' + df['Genre'] + ' ' + df['Publisher']
    return df
df = preprocess_data(df)
df.head()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_model(df):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim
cosine_sim = build_model(df)
joblib.dump(cosine_sim, 'model.pkl')


def get_recommendations(title, df, cosine_sim):
    idx = df[df['Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    book_indices = [i[0] for i in sim_scores]
    return df['Title'].iloc[book_indices]

username = input("Enter book:")
print(get_recommendations(username, df, cosine_sim))



