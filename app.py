from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
CV = pickle.load(open('transform.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
def home():
   return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        data = [text]
        vect = CV.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('index.html', prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
