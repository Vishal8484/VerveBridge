from flask import Flask, request, render_template
import pandas as pd
import joblib


app = Flask(__name__)

# joblib is use for thr load the model
cosine_sim = joblib.load('model.pkl')


df = pd.read_csv('books.csv')


def preprocess_data(df):
    df.fillna('', inplace=True)
    df['combined_features'] = df['Title'] + ' ' + df['Author'] + ' ' + df['Genre'] + ' ' + df['Publisher']
    return df

df = preprocess_data(df)


def get_recommendations(title, df, cosine_sim):
    if title not in df['Title'].values:
        return ["Book not found"]
    
    idx = df[df['Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11] 
    
    valid_indices = [i[0] for i in sim_scores if i[0] < len(df)]
    
    return df['Title'].iloc[valid_indices].tolist()

    unique_book_indices = []
    seen_titles = set()
    for i, score in sim_scores:
        if len(unique_book_indices) == 10:
            break
        book_title = df['Title'].iloc[i]
        if book_title != title and book_title not in seen_titles:
            unique_book_indices.append(i)
            seen_titles.add(book_title)

    
    return df['Title'].iloc[unique_book_indices].tolist()


@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    if request.method == 'POST':
        title = request.form['title']
        recommendations = get_recommendations(title, df, cosine_sim)
    return render_template('index.html', recommendations=recommendations)
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
